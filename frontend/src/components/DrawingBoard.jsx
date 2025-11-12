import { useEffect, useRef, useState, useCallback } from "react";
import "./DrawingBoard.css";

const CANVAS_SIZE = 280;
const STROKE_STYLE = "#ffffff";
const LINE_WIDTH = 22;

const DrawingBoard = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = STROKE_STYLE;
    ctx.lineWidth = LINE_WIDTH;
  }, []);

  const getCanvasCoordinates = useCallback((event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }, []);

  const startDrawing = useCallback(
    (event) => {
      const ctx = canvasRef.current.getContext("2d");
      const { x, y } = getCanvasCoordinates(event);

      ctx.beginPath();
      ctx.moveTo(x, y);
      setIsDrawing(true);
    },
    [getCanvasCoordinates],
  );

  const draw = useCallback(
    (event) => {
      if (!isDrawing) {
        return;
      }

      const ctx = canvasRef.current.getContext("2d");
      const { x, y } = getCanvasCoordinates(event);

      ctx.lineTo(x, y);
      ctx.stroke();
    },
    [getCanvasCoordinates, isDrawing],
  );

  const stopDrawing = useCallback(() => {
    if (!isDrawing) {
      return;
    }

    const ctx = canvasRef.current.getContext("2d");
    ctx.closePath();
    setIsDrawing(false);
  }, [isDrawing]);

  const handleClear = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    setPrediction(null);
  }, []);

  const handlePredict = useCallback(async () => {
    const canvas = canvasRef.current;
    const image = canvas.toDataURL("image/png");
    console.log(canvas.toDataURL('image/png'))
    try {
      const response = await fetch("http://127.0.0.1:9000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image_base64: image }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result.prediction);
    } catch (error) {
      console.error(error);
      setPrediction("Không thể nhận diện – kiểm tra lại backend");
    }
  }, []);

  return (
    <div className="drawing-board">
      <h1>MNIST Handwriting Recognition</h1>
      <p>Vẽ chữ số (0-9) bằng chuột hoặc trackpad trên bảng bên dưới.</p>

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
          <span>Hãy vẽ một số để nhận diện.</span>
        ) : (
          <span>
            Kết quả nhận diện:
            <strong>{` ${prediction}`}</strong>
          </span>
        )}
      </div>
    </div>
  );
};

export default DrawingBoard;


