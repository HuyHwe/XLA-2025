import { useEffect, useRef, useState, useCallback } from "react";
import "../styles/DrawingBoard.css";
import ModelSelector from "./ModelSelector";
import PreprocessPanel from "./PreprocessPanel";

// Các hằng số cấu hình canvas
const CANVAS_SIZE = 280;  // Kích thước canvas (280x280 pixels)
const STROKE_STYLE = "#ffffff";  // Màu nét vẽ (trắng)
const LINE_WIDTH = 22;  // Độ dày nét vẽ
const API_BASE_URL = "http://127.0.0.1:9000";  // URL backend API

const DrawingBoard = () => {
  // Refs và State
  const canvasRef = useRef(null);  // Reference đến canvas element
  const [isDrawing, setIsDrawing] = useState(false);  // Trạng thái đang vẽ
  const [prediction, setPrediction] = useState(null);  // Kết quả dự đoán
  const [modelType, setModelType] = useState("number_custom");  // Model đang chọn
  const [availableModels, setAvailableModels] = useState({});  // Danh sách models từ backend
  const [isLoadingModels, setIsLoadingModels] = useState(true);  // Trạng thái đang load models
  const [preprocessSteps, setPreprocessSteps] = useState([]); // Danh sách ảnh từng bước tiền xử lý

  // Khởi tạo canvas khi component mount
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Thiết lập kích thước canvas
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    // Vẽ nền đen
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Cấu hình style vẽ
    ctx.lineCap = "round";  // Đầu nét vẽ tròn
    ctx.lineJoin = "round";  // Góc nối tròn
    ctx.strokeStyle = STROKE_STYLE;  // Màu trắng
    ctx.lineWidth = LINE_WIDTH;  // Độ dày nét
  }, []);

  // Load danh sách models từ backend khi component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        // Gọi API để lấy danh sách models
        const response = await fetch(`${API_BASE_URL}/models`);
        if (response.ok) {
          const data = await response.json();
          setAvailableModels(data.models);
          
          // Tự động chọn model đầu tiên có sẵn làm mặc định
          const available = Object.keys(data.models).filter(
            key => data.models[key].available
          );
          if (available.length > 0) {
            setModelType(available[0]);
          }
        }
      } catch (error) {
        console.error("Failed to fetch models:", error);
      } finally {
        setIsLoadingModels(false);  // Đánh dấu đã load xong
      }
    };

    fetchModels();
  }, []);

  // Chuyển đổi tọa độ chuột sang tọa độ canvas (xử lý responsive)
  const getCanvasCoordinates = useCallback((event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();  // Lấy vị trí và kích thước canvas trên màn hình
    
    // Tính tỷ lệ scale giữa kích thước thực và kích thước hiển thị
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Chuyển đổi tọa độ chuột sang tọa độ canvas
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }, []);

  // Bắt đầu vẽ khi nhấn chuột
  const startDrawing = useCallback(
    (event) => {
      const ctx = canvasRef.current.getContext("2d");
      const { x, y } = getCanvasCoordinates(event);

      // Bắt đầu đường vẽ mới tại vị trí chuột
      ctx.beginPath();
      ctx.moveTo(x, y);
      setIsDrawing(true);  // Đánh dấu đang vẽ
    },
    [getCanvasCoordinates],
  );

  // Vẽ khi di chuyển chuột (khi đang nhấn)
  const draw = useCallback(
    (event) => {
      if (!isDrawing) {
        return;  // Không vẽ nếu chưa bắt đầu
      }

      const ctx = canvasRef.current.getContext("2d");
      const { x, y } = getCanvasCoordinates(event);

      // Vẽ đường thẳng đến vị trí mới
      ctx.lineTo(x, y);
      ctx.stroke();
    },
    [getCanvasCoordinates, isDrawing],
  );

  // Dừng vẽ khi thả chuột
  const stopDrawing = useCallback(() => {
    if (!isDrawing) {
      return;
    }

    const ctx = canvasRef.current.getContext("2d");
    ctx.closePath();  // Đóng đường vẽ
    setIsDrawing(false);  // Đánh dấu đã dừng vẽ
  }, [isDrawing]);

  // Xóa canvas và reset kết quả dự đoán
  const handleClear = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Vẽ lại nền đen để xóa
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    setPrediction(null);  // Reset kết quả
    setPreprocessSteps([]); // Ẩn khung tiền xử lý
  }, []);

  // Gửi ảnh lên backend để nhận diện
  const handlePredict = useCallback(async () => {
    const canvas = canvasRef.current;
    // Chuyển canvas thành base64 string (PNG format)
    const image = canvas.toDataURL("image/png");
    
    try {
      // Gửi request đến API với ảnh và model type
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          image_base64: image,  // Ảnh dạng base64
          model_type: modelType,  // Model đã chọn
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      // Nhận kết quả dự đoán từ backend
      const result = await response.json();
      setPrediction(result.prediction);
      setPreprocessSteps(result.preprocessing_steps || []); // Lưu các bước tiền xử lý (nếu có)
    } catch (error) {
      console.error(error);
      setPrediction("Không thể nhận diện – kiểm tra lại backend");
    }
  }, [modelType]);

  // Lấy thông tin model hiện tại
  const currentModel = availableModels[modelType];
  const getInstructionText = () => {
    if (!currentModel) return "Đang tải...";
    if (currentModel.category === "number") {
      return "Vẽ chữ số (0-9) bằng chuột hoặc trackpad trên bảng bên dưới.";
    } else if (currentModel.category === "shape") {
      return "Vẽ hình dạng (Tròn, Tam giác, Hình thoi, Hình vuông) bằng chuột hoặc trackpad trên bảng bên dưới.";
    } else if (currentModel.category === "char") {
      return "Vẽ chữ cái (A-Z) bằng chuột hoặc trackpad trên bảng bên dưới.";
    }
    return "Vẽ trên bảng bên dưới.";
  };

  const hasPreprocessSteps = preprocessSteps && preprocessSteps.length > 0;

  return (
    <div className={`drawing-page ${hasPreprocessSteps ? "drawing-page--with-steps" : ""}`}>
      <div className="drawing-page__left">
        <div className="drawing-board">
          <h1>Nhận diện Vẽ tay</h1>
          <p>{getInstructionText()}</p>

          {/* Dropdown chọn model để sử dụng */}
          <ModelSelector
            modelType={modelType}
            availableModels={availableModels}
            isLoading={isLoadingModels}
            onChange={setModelType}
          />

          <canvas
            ref={canvasRef}
            className="drawing-board__canvas"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
          />

          <div className="drawing-board__actions">
            <button type="button" onClick={handlePredict}>
              Nhận diện
            </button>
            <button type="button" className="button-secondary" onClick={handleClear}>
              Xóa
            </button>
          </div>

          <div className="drawing-board__result">
            {prediction === null ? (
              <span>
                {currentModel?.category === "number" && "Hãy vẽ một số để nhận diện."}
                {currentModel?.category === "shape" && "Hãy vẽ một hình dạng để nhận diện."}
                {currentModel?.category === "char" && "Hãy vẽ một chữ cái để nhận diện."}
                {!currentModel && "Đang tải..."}
              </span>
            ) : (
              <span>
                Kết quả nhận diện:
                <strong>{` ${prediction}`}</strong>
                <br />
                {currentModel && (
                  <span className="result__model-info">
                    {" "}(Model: {currentModel.name})
                  </span>
                )}
              </span>
            )}
          </div>
        </div>
      </div>

      {hasPreprocessSteps && <PreprocessPanel steps={preprocessSteps} />}
    </div>
  );
};

export default DrawingBoard;


