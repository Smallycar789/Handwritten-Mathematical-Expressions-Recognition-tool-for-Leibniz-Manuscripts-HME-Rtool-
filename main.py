import sys, os, time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem, QCheckBox, QGraphicsPolygonItem, QGraphicsTextItem, QMessageBox, QDialog, QListWidget
)
from PyQt5.QtGui import QPixmap, QPolygonF, QPainterPath, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPointF
from PIL import Image
from processor import recognize
import cv2
import json

class PolygonGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None
        self.polygon_points = []
        self.point_items = []   # 显示的小圆点
        self.line_items = []    # 显示的线条
        self.polygons_data = [] # 存储每个多边形的数据
        self._zoom = 0
        self._pan = False
        self._pan_start = None
        self.polygon_items = []  # 存储显示的多边形和文本项}
        

    def load_image(self, path):
        pixmap = QPixmap(path)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.image_item.setZValue(0)
        self.scene.clear()
        self.scene.addItem(self.image_item)

        # 重置
        self.polygon_points = []
        self.point_items = []
        self.line_items = []
        self.polygons_data = []
        self._zoom = 0
        self.resetTransform()

    def wheelEvent(self, event):
        """Zoom centré sur la souris"""
        if not self.image_item:
            return
        if event.angleDelta().y() > 0:
            factor = 1.25
        else:
            factor = 0.8
        old_pos = self.mapToScene(event.pos())
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event):
        if not self.image_item:
            return

        pos = self.mapToScene(event.pos())

        if event.button() == Qt.LeftButton:
            # click the first point to close the polygon
            if self.polygon_points and (pos - self.polygon_points[0]).manhattanLength() < 10:
                self.finish_polygon()
                return

            # add point
            self.polygon_points.append(pos)

            # show small circle 
            circ = QGraphicsEllipseItem(pos.x()-5, pos.y()-5, 10, 10)
            circ.setBrush(QColor("red"))
            circ.setPen(QPen(Qt.NoPen))
            circ.setZValue(1)
            self.scene.addItem(circ)
            self.point_items.append(circ)

            # add line
            if len(self.polygon_points) > 1:
                line = QGraphicsLineItem(
                    self.polygon_points[-2].x(), self.polygon_points[-2].y(),
                    self.polygon_points[-1].x(), self.polygon_points[-1].y()
                )
                line.setPen(QPen(QColor("red"), 4))
                line.setZValue(1)
                self.scene.addItem(line)
                self.line_items.append(line)

        elif event.button() == Qt.RightButton:
            # right-click to close the polygon
            self.finish_polygon()

    def finish_polygon(self):
        """close the polygon"""
        if len(self.polygon_points) < 3:
            return

        # close the last line
        line = QGraphicsLineItem(
            self.polygon_points[-1].x(), self.polygon_points[-1].y(),
            self.polygon_points[0].x(), self.polygon_points[0].y()
        )
        line.setPen(QPen(QColor("red"), 4))
        line.setZValue(1)
        self.scene.addItem(line)
        self.line_items.append(line)


    def reset_polygon(self):
        """clear the current polygon"""
        for item in self.point_items + self.line_items:
            if item.scene() == self.scene:
                self.scene.removeItem(item)
        self.polygon_points = []
        self.point_items = []
        self.line_items = []

    def get_cropped_image(self):
        """return the cropped PIL image with a white background"""
        if not self.image_item or len(self.polygon_points) < 3:
            return None

        pixmap = self.image_item.pixmap()
        image = pixmap.toImage()
        w, h = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()  # BGRA
        arr = arr[..., :3]  # BGR
        arr = arr[..., ::-1]  # BGR -> RGB

        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[int(p.x()), int(p.y())] for p in self.polygon_points], np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # background white
        result = np.ones_like(arr, dtype=np.uint8) * 255
        result[mask == 255] = arr[mask == 255]

        # Crop bounding box
        x, y, w_box, h_box = cv2.boundingRect(pts)
        cropped = result[y:y+h_box, x:x+w_box]

        return Image.fromarray(cropped)

    def save_polygon_data(self, image_name, recognition_result):
        """save the current polygon's coordinates and recognition result"""
        points_list = [[float(p.x()), float(p.y())] for p in self.polygon_points]
        self.polygons_data.append({
            "image": image_name,
            "points": points_list,
            "result": recognition_result
        })

    def show_polygon(self):
        if not self.polygons_data:
            return
        
        last_result = self.polygons_data[-1]
        result_text = last_result.get("result", "")
        pts = [QPointF(x, y) for x, y in last_result["points"]]

        polygon_item = QGraphicsPolygonItem(QPolygonF(pts))
        polygon_item.setPen(QPen(Qt.green, 4))
        polygon_item.setBrush(QColor(0, 255, 0, 50))
        self.scene.addItem(polygon_item)

        min_x = min(p.x() for p in pts)
        min_y = min(p.y() for p in pts) - 20

        text_item = QGraphicsTextItem(result_text)
        text_item.setDefaultTextColor(Qt.white) 
        text_item.setPos(min_x, min_y)
        self.scene.addItem(text_item)

        rect = text_item.boundingRect()
        bg_rect_item = self.scene.addRect(
            min_x, min_y, rect.width(), rect.height(),
            pen=QPen(Qt.NoPen),
            brush=QColor("blue")
        )
        text_item.setZValue(bg_rect_item.zValue() + 1)

        if not hasattr(self, "polygon_items"):
            self.polygon_items = []
        
        self.polygon_items.append({
            "polygon": polygon_item,
            "text": text_item,
            "bg": bg_rect_item
        })
    


    def show_all_polygons(self):
        if not hasattr(self, "polygon_items"):
            self.polygon_items = []

        for item in self.polygon_items:
            self.scene.removeItem(item["polygon"])
            self.scene.removeItem(item["text"])
            self.scene.removeItem(item["bg"])
        self.polygon_items.clear()

        for poly_data in self.polygons_data:
            result_text = poly_data.get("result", "")
            pts = [QPointF(x, y) for x, y in poly_data["points"]]
            polygon_item = QGraphicsPolygonItem(QPolygonF(pts))
            polygon_item.setPen(QPen(Qt.green, 4))
            polygon_item.setBrush(QColor(0, 255, 0, 50))
            self.scene.addItem(polygon_item)

            min_x = min(p.x() for p in pts)
            min_y = min(p.y() for p in pts) - 20


            text_item = QGraphicsTextItem(result_text)
            text_item.setDefaultTextColor(Qt.white)
            text_item.setPos(min_x, min_y)
            self.scene.addItem(text_item)

            rect = text_item.boundingRect()
            bg_rect_item = self.scene.addRect(
                min_x, min_y, rect.width(), rect.height(),
                pen=QPen(Qt.NoPen),
                brush=QColor("blue")
            )

            text_item.setZValue(bg_rect_item.zValue() + 1)


            self.polygon_items.append({
                "polygon": polygon_item,
                "text": text_item,
                "bg": bg_rect_item
            })


    def update_polygon_text(self, idx, new_text):
        if idx < 0 or idx >= len(self.polygon_items):
            return
        
        item_dict = self.polygon_items[idx]
        text_item = item_dict["text"]
        bg_item = item_dict["bg"]

        text_item.setPlainText(new_text)

        rect = text_item.boundingRect()
        bg_item.setRect(text_item.x(), text_item.y(), rect.width(), rect.height())


    def delete_polygon(self, idx):
        if idx < 0 or idx >= len(self.polygon_items):
            return
        
        item_dict = self.polygon_items.pop(idx)

        for key in ["polygon", "text", "bg"]:
            item = item_dict[key]
            if item.scene() is not None:
                self.scene.removeItem(item)
    
        if idx < len(self.polygons_data):
            self.polygons_data.pop(idx)


    def get_cropped_image_by_index(self, idx):
        """Return cropped PIL image for an existing polygon in polygons_data"""
        if self.image_item is None:
            return None
        if idx < 0 or idx >= len(self.polygons_data):
            return None

        pixmap = self.image_item.pixmap()
        image = pixmap.toImage()
        w, h = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()  # BGRA
        arr = arr[..., :3]  # BGR
        arr = arr[..., ::-1]  # BGR -> RGB

        # polygon points
        pts = np.array([[int(x), int(y)] for x, y in self.polygons_data[idx]["points"]], np.int32)
        if len(pts) < 3:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # background white
        result = np.ones_like(arr, dtype=np.uint8) * 255
        result[mask == 255] = arr[mask == 255]

        # Crop bounding box
        x, y, w_box, h_box = cv2.boundingRect(pts)
        cropped = result[y:y+h_box, x:x+w_box]

        return Image.fromarray(cropped)


class HME_Rtool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HME-Rtool")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)


        # image view
        self.view = PolygonGraphicsView()
        layout.addWidget(self.view, stretch=3)

        # right panel
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)

        # operation tips
        self.label_tips1 = QLabel("Reset the polygon (Esc)  Recognize (Enter)")
        self.label_tips1.setStyleSheet("color: black; font-weight: bold;")
        right_panel.insertWidget(0, self.label_tips1)  # Insert at the top

        self.save_crop_auto_checkbox = QCheckBox("save cropped images auto")
        right_panel.addWidget(self.save_crop_auto_checkbox)

        self.btn_load = QPushButton("Import Image")
        self.btn_load.clicked.connect(self.load_image)
        right_panel.addWidget(self.btn_load)

        self.btn_recognize = QPushButton("Recognize")
        self.btn_recognize.clicked.connect(self.process_polygon)
        right_panel.addWidget(self.btn_recognize)


        self.label_tips3 = QLabel("Recognition Result")
        self.label_tips3.setStyleSheet("color: black; font-weight: bold;")
        right_panel.addWidget(self.label_tips3)

        self.text_result = QTextEdit()
        self.text_result.setPlaceholderText("Waiting for recognition...")
        self.text_result.setFixedHeight(100)
        right_panel.addWidget(self.text_result)

        self.crop_label = QLabel("Expression image preview here")
        self.crop_label.setFixedSize(480, 220)
        self.crop_label.setStyleSheet("background: white;")
        right_panel.addWidget(self.crop_label)

        self.label_tips2 = QLabel("Polygons Management")
        self.label_tips2.setStyleSheet("color: black; font-weight: bold;")
        right_panel.addWidget(self.label_tips2)

        self.current_selected_idx = -1
        self.polygon_list = QListWidget()
        right_panel.addWidget(self.polygon_list)

        self.label_tips4 = QLabel("Rewrite Expression")
        self.label_tips4.setStyleSheet("color: black; ")
        right_panel.addWidget(self.label_tips4)

        self.rewrite_text = QTextEdit()
        self.rewrite_text.setPlaceholderText("Rewrite selected polygon result...")
        self.rewrite_text.setFixedHeight(100)
        right_panel.addWidget(self.rewrite_text)

        self.btn_confirm_rewrite = QPushButton("Confirm Rewrite Selected Polygon")
        self.btn_confirm_rewrite.clicked.connect(self.rewrite_selected_polygon)
        right_panel.addWidget(self.btn_confirm_rewrite)

        self.delete_btn = QPushButton("Delete Selected Polygon")
        self.delete_btn.clicked.connect(self.delete_selected_polygon)
        right_panel.addWidget(self.delete_btn)

        self.polygon_list.currentRowChanged.connect(self.on_polygon_selected)


    def on_polygon_selected(self, idx):
        self.current_selected_idx = idx
        if idx < 0 or idx >= len(self.view.polygons_data):
            self.rewrite_text.clear()
            return
        self.rewrite_text.setPlainText(self.view.polygons_data[idx]['result'])

        # show cropped preview
        cropped = self.view.get_cropped_image_by_index(idx)
        if cropped is not None:
            pixmap = QPixmap.fromImage(self.pil2qimage(cropped))
            pixmap = pixmap.scaled(
                self.crop_label.width(),
                self.crop_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.crop_label.setPixmap(pixmap)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.view.reset_polygon()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.process_polygon()
        
        super().keyPressEvent(event)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.view.load_image(path)
        self.current_image_path = path
        self.load_json_results()
        self.refresh_polygon_list()

    def save_crop_auto(self, cropped):
        folder = "crops"
        os.makedirs(folder, exist_ok=True)

        # get the first point coordinates
        if not self.view.polygon_points:
            return
        x0 = int(self.view.polygon_points[0].x())
        y0 = int(self.view.polygon_points[0].y())


        # original image name
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]

        # concatenate file name
        filename = f"{base_name}_{x0}_{y0}.png"
        filepath = os.path.join(folder, filename)

        cropped.save(filepath)
        print(f"Saved cropped image: {filepath}")

    def load_json_results(self):
        if not hasattr(self, "current_image_path"):
            return
        image_name = os.path.basename(self.current_image_path)
        folder = "results"
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{image_name[:-4]}.json")

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.view.polygons_data = data.get("polygons", [])
                self.view.show_all_polygons()


    

    def pil2qimage(self, im):
        im = im.convert("RGB")
        data = im.tobytes("raw", "RGB")
        qimage = QImage(
            data, 
            im.width, 
            im.height, 
            im.width * 3,  # 每行字节数 = 宽度 × 3 (RGB)
            QImage.Format_RGB888
        )
        return qimage



    def save_json_result(self):
        if not self.current_image_path:
            return
        folder = "results"
        os.makedirs(folder, exist_ok=True)
        image_name = os.path.basename(self.current_image_path)
        filepath = os.path.join(folder, f"{image_name[:-4]}.json")

        # if json exists
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"image": image_name, "polygons": []}

        # update polygons_data in memory
        data["polygons"] = self.view.polygons_data

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


    def process_polygon(self):
        cropped = self.view.get_cropped_image()
        if cropped is None:
            return

        # recognize
        result = recognize(cropped, version="28")

        # save the polygon data in memory
        self.view.save_polygon_data(os.path.basename(self.current_image_path), result)

        # cropped image preview
        qimg = self.pil2qimage(cropped)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.crop_label.width(),
            self.crop_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.crop_label.setPixmap(pixmap)


        # show recognition result
        self.text_result.setText(result)

        # save cropped image if checkbox is checked
        if self.save_crop_auto_checkbox.isChecked():
            self.save_crop_auto(cropped)

        # show polygon
        self.view.show_polygon()

        # save all polygon data to JSON
        self.save_json_result()

        self.refresh_polygon_list()

        # reset current polygon
        self.view.reset_polygon()


    def refresh_polygon_list(self):

        self.polygon_list.clear()
        for i, poly in enumerate(self.view.polygons_data):
            text = f"Polygon {i+1}: {poly.get('result','')}"
            self.polygon_list.addItem(text)

    def delete_selected_polygon(self):
        idx = self.current_selected_idx
        if idx < 0:
            return
        self.view.delete_polygon(idx)
        self.refresh_polygon_list()
        self.save_json_result()
        self.current_selected_idx = -1 

    def rewrite_selected_polygon(self):
        idx = self.polygon_list.currentRow()
        if idx < 0:
            return
        new_text = self.rewrite_text.toPlainText()
        self.view.polygons_data[idx]['result'] = new_text

        self.view.update_polygon_text(idx, new_text)

        self.refresh_polygon_list()
        self.save_json_result()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HME_Rtool()
    window.show()
    sys.exit(app.exec_())

