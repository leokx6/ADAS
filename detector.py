import torch
from PIL import Image, ImageDraw
import random
import math
from scipy.interpolate import splev, splprep
import numpy as np
import cv2
import os
import time

def generate_binary_mask(img, save_mask_path, line_color=(0,255,255)):
    img_array = np.array(img)
    color_diff = np.abs(img_array - np.array(line_color))
    mask = np.all(color_diff <= 0, axis=2).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")
    mask_img.save(save_mask_path)

def order_img(img_dir):
    imgs = []
    for img_name in os.listdir(img_dir):
        imgs.append(img_name)
    imgs.sort()

    return imgs

def max_box(boxes):
    max_height = 0
    max_box = 0
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i][:4]
        if abs(y2 - y1) > max_height:
            max_height = abs(y2 - y1)
            max_box = i
    return max_box


def lowest_point(points):
    lowest = 0
    for i in range(len(points)):
        if points[i][1] > points[lowest][1]:
            lowest = i
    return lowest


 
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def list_graphs(boxes, points, class_ids):
    graphs = []
    boxes_classified = []
    points_classified = []
    for i in range(9):
        boxes_class = []
        mid_points_class = []
        for j in range(len(boxes)):
            if class_ids[j] == i:
                boxes_class.append(boxes[j])
                mid_points_class.append(points[j])
        graph = graph_generator(boxes_class, mid_points_class)
        graphs.append(graph)
        boxes_classified.append(boxes_class)
        points_classified.append(mid_points_class)
    return graphs, boxes_classified, points_classified


def graph_generator(boxes, points):
    graph = []
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                height1 = boxes[i][3] - boxes[i][1]
                height2 = boxes[j][3] - boxes[j][1]
                distance = (((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 3) ** 0.5 ) * (1 / height1 + 1 / height2)
                graph.append((i, j, distance))
    graph.sort(key=lambda x: x[2])
    return graph

def inference(model, imgs):
    # Inference
    results = model(imgs)

    # Dizionario dei colori definiti manualmente per ciascuna classe
    # Ogni classe è associata a un colore RGB
    CLASS_COLORS = {
        0: (255, 0, 0),    # Rosso
        1: (0, 255, 0),    # Verde
        2: (0, 0, 255),    # Blu
        3: (255, 255, 0),  # Giallo
        4: (255, 165, 0),  # Arancione
        5: (128, 0, 128),  # Viola
        6: (0, 255, 255),  # Azzurro
        7: (255, 192, 203),# Rosa
        8: (128, 128, 0)   # Oliva
    }

    # Funzione per disegnare bounding box con colori specifici per classe
    def draw_colored_boxes(img, boxes, class_ids):
        draw = ImageDraw.Draw(img)
        for box, cls_id in zip(boxes, class_ids):
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))  # Default: bianco se la classe non è nel dizionario
            draw.rectangle(box[:4], outline=color, width=3)
        return img
    
    def draw_colored_points(img, mid_points, class_ids):
        draw = ImageDraw.Draw(img)
        for point, cls_id in zip(mid_points, class_ids):
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))  # Default: bianco se la classe non è nel dizionario
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
        return img

    '''def isoutlier(boxes, i, j, x1, x2, y1, y2):
        box_area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        box_area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
        print(box_area_i, box_area_j)
        pixel_distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        print(pixel_distance)
        if pixel_distance > 500 and box_area_i > 3000 and box_area_j > 3000:
            return False
        elif pixel_distance < 500:
            return False
        else:
            return True'''

    def isoutlier(x1, x2, x3, y1, y2, y3):
        # Se l'angolo tra i due segmenti è maggiore di 90 gradi, ritorna True
        angle = getAngle((x1, y1), (x2, y2), (x3, y3)) - 180
        print("Angle: ", angle)
        if abs(angle) > 160:
            return True
        else:
            return False
        

    def draw_colored_lines(img, points_classified, boxes_classified, graphs):
        draw = ImageDraw.Draw(img)
        for class_id in range(9):
            if len(boxes_classified[class_id]) == 0:
                continue
            if class_id != 2 and class_id != 6:
                continue
            graph = graphs[class_id]
            boxes = boxes_classified[class_id]
            mid_points = points_classified[class_id]
            # Prendi l'indice del punto più basso per la classe corrente
            starting_index = lowest_point(mid_points)
            connected = [starting_index]
            current_index = starting_index
            last_index = starting_index
            while len(connected) < len(boxes):
                for i, j, distance in graph:
                    if i == current_index and j not in connected:
                        x2, y2 = mid_points[i]
                        x3, y3 = mid_points[j]
                        x1, y1 = mid_points[last_index]
                        color = CLASS_COLORS.get(class_id, (255, 255, 255))
                        if i == last_index or not isoutlier(x1, x2, x3, y1, y2, y3):
                            print("Drawing line between ", i, " and ", j)
                            draw.line([mid_points[i], mid_points[j]], fill=color, width=3)
                        connected.append(j)
                        current_index = j
                        last_index = i
                        break
        return img
    
    def draw_colored_curves(img, points_classified, boxes_classified, graphs):
        draw = ImageDraw.Draw(img)
        for class_id in range(9):
            if len(boxes_classified[class_id]) == 0:
                continue
            if class_id != 2 and class_id != 6:
                continue
            graph = graphs[class_id]
            boxes = boxes_classified[class_id]
            mid_points = points_classified[class_id]
            # Prendi l'indice della box più alta per la classe corrente
            starting_index = lowest_point(mid_points)
            connected = [starting_index]
            ordered_points = [mid_points[starting_index]]
            current_index = starting_index
            while True:
                for i, j, distance in graph:
                    if i == current_index and j not in connected:
                        ordered_points.append(mid_points[j])
                        connected.append(j)
                        current_index = j
                        break
                if len(connected) == len(boxes):
                    break
            # Polinomial interpolation of the points to draw a curve
            npImg = np.array(img)
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            # Use Centripetal Catmull–Rom spline
            tck, u = splprep(np.array(ordered_points).T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)
            ordered_points = np.column_stack((x_new, y_new)).tolist()
            cv2.polylines(npImg, [np.array(ordered_points).astype(int)], isClosed=False, color=color, thickness=3)
            img = Image.fromarray(npImg)
        return img

    
    # Funzione per calcolare l'Intersection over Union (IoU) tra due bounding box
    def iou(box1, box2):
        # Calcola l'area di intersezione
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        # Calcola l'area dell'unione
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area_box1 + area_box2 - intersection_area
        # Calcola e restituisce l'IoU
        return intersection_area / union_area

    # Esegui inferenza
    results = model(imgs)

    # Ottieni bounding box e classi
    boxes = results.xyxy[0].cpu().numpy()  # Bounding box: [xmin, ymin, xmax, ymax, conf, cls]
    class_ids = boxes[:, 5].astype(int)    # Indici di classe predetti
    img_path = imgs[0]

    # Leggi fiducia e classe predetta
    confidences = boxes[:, 4]  # Fiducia
    # Se due boxes si sovrappongono per più del 60%, mantieni solo quella con fiducia maggiore
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i], boxes[j]) > 0.6:
                if confidences[i] > confidences[j]:
                    boxes[j] = 0
                else:
                    boxes[i] = 0
    boxes = boxes[boxes[:, 0] != 0]
    confidences = boxes[:, 4]

    # Rimuovi bounding box con fiducia minore di 0.5
    boxes = boxes[confidences > 0.5]


    # Rimuovi bounding box con area minore di 200 pixel
    boxes = boxes[(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 200]


    # Trova punti medi sulla base delle coordinate dei bounding box
    mid_points = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        mid_points.append(((x1 + x2) / 2, y2))

    # genera lista di grafi dove in ogni grafo ci sono i punti medi e le distanze tra i punti di una classe
    #Misura tempo di esecuzione
    start = time.time()
    graphs, boxes_classified, points_classified = list_graphs(boxes, mid_points, class_ids)
    end1 = time.time()

    img = Image.open(img_path)

    # Disegna bounding box colorati in base alle classi
    #img_with_colored_boxes = draw_colored_boxes(img, boxes, class_ids)

    #img_with_colored_boxes = draw_colored_points(img, mid_points, class_ids)

    img_with_colored_boxes = draw_colored_lines(img, points_classified, boxes_classified, graphs)
    mask = generate_binary_mask(img_with_colored_boxes, "D:/Desktop/Adas_test/test.jpg")
    end2 = time.time()
    print("Time for graph generation: ", end1 - start)
    print("Time for drawing lines: ", end2 - end1)

    # Salva il risultato
    img_with_colored_boxes.save('output_colored_boxes_by_class.jpg')

    # Visualizza il risultato con opencv
    img_cv = cv2.imread('output_colored_boxes_by_class.jpg')
    resized = cv2.resize(img_cv, (800, 600), interpolation=cv2.INTER_AREA)
    cv2.imshow('Result', resized)
    cv2.waitKey(0)

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load custom model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolom.pt')

# Cicla sulle immagini jpg nella cartella Dataset/amz/img
img_dir = 'Dataset/ugr/img'
imgs = order_img(img_dir)

for img_name in imgs:
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        img_path = os.path.join(img_dir, img_name)
        start = time.time()
        inference(model, [img_path])
        end = time.time()
        print("Time for inference: ", end - start)

