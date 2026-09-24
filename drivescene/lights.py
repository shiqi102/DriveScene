from pathlib import Path


class YOLOv8Lights:
    def __init__(self, checkpoint, state_classes, device="cpu"):
        from ultralytics import YOLO
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        self.model = YOLO(str(checkpoint))
        self.state_classes = state_classes
        self.device = device

    def predict(self, cameras):
        results = self.model.predict([c["path"] for c in cameras], device=self.device, verbose=False)
        lights = []
        for camera, result in zip(cameras, results):
            for box in result.boxes:
                name = result.names[int(box.cls.item())]
                if name not in self.state_classes and name != "traffic light":
                    continue
                lights.append({"id": f"{camera['name']}:{len(lights)}", "camera": camera["name"],
                               "bbox_xyxy": box.xyxy[0].tolist(), "score": float(box.conf.item()),
                               "state": self.state_classes.get(name, "unknown"), "position_ego": None,
                               "state_source": "trained_state_class" if name in self.state_classes else "unavailable"})
        return lights
