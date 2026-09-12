# PPE Detection Edge System

Real-time PPE (helmet) detection system on edge device (Jetson Nano) with full MLOps pipeline: YOLOv8 → TensorRT → ASP.NET Core → Blazor dashboard.

## Architecture
[diagramme à ajouter]

## Tech Stack
- **ML**: YOLOv8 (Ultralytics), TensorRT
- **Backend**: ASP.NET Core, SignalR, PostgreSQL
- **Frontend**: Blazor WebAssembly
- **Deployment**: Jetson Nano, Docker

## Status
🚧 In progress
- [x] Dataset preparation (Roboflow, 7k images)
- [x] Model training (YOLOv8n)
- [ ] TensorRT optimization
- [x] Backend API
- [ ] Blazor dashboard
- [ ] Docker deployment
