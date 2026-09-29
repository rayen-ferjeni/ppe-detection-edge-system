import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("../models/ppe.engine")
print("Moteur chargé avec succès !")
def preprocess(image_path, input_size=640):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_transposed, axis=0)
    return np.ascontiguousarray(img_batch)


def allocate_buffers(engine):
    context = engine.create_execution_context()
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)

    input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
    output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    print("Mémoire GPU allouée")
    return context, d_input, d_output, output_shape
def apply_nms(boxes, scores, class_ids, iou_threshold=0.5):
    if len(boxes) == 0:
        return [], [], []

    boxes_array = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_array, scores, score_threshold=0.0, nms_threshold=iou_threshold)

    final_boxes = []
    final_scores = []
    final_class_ids = []

    for i in indices:
        i = int(i)
        final_boxes.append(boxes[i])
        final_scores.append(scores[i])
        final_class_ids.append(class_ids[i])

    return final_boxes, final_scores, final_class_ids
def infer(context, d_input, d_output, input_data, output_shape):
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    return output_data

def postprocess(output, conf_threshold=0.5, class_names=["head", "helmet", "person"]):
    predictions = output[0].T  # (8400, 7)

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence >= conf_threshold:
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            x = cx - w / 2
            y = cy - h / 2
            boxes.append([x, y, w, h])
            scores.append(float(confidence))
            class_ids.append(class_id)

    return boxes, scores, class_ids
input_data = preprocess("../test_images/test.jpg")
print("Image préparée, shape :", input_data.shape)

context, d_input, d_output, output_shape = allocate_buffers(engine)
result = infer(context, d_input, d_output, input_data, output_shape)

print("Inférence terminée !")
predictions = result[0].T
all_scores = predictions[:, 4:]

print("Score de confiance maximum trouvé :", np.max(all_scores))
print("Score de confiance moyen :", np.mean(all_scores))
boxes, scores, class_ids = postprocess(result, conf_threshold=0.3 )
boxes, scores, class_ids = apply_nms(boxes, scores, class_ids)
class_names = ["head", "helmet", "person"]
print(f"Nombre de détections après filtrage + NMS : {len(boxes)}")
for box, score, class_id in zip(boxes, scores, class_ids):
    print(f"- {class_names[class_id]} détecté avec {score:.2f} de confiance, position: {box}")
