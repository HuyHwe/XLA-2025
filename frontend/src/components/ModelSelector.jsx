import PropTypes from "prop-types";

const ModelSelector = ({ modelType, availableModels, isLoading, onChange }) => (
  <div className="drawing-board__model-selector">
    <label htmlFor="model-select" className="model-selector__label">
      Chọn mô hình:
    </label>
    <select
      id="model-select"
      value={modelType}
      onChange={(e) => onChange(e.target.value)}
      className="model-selector__select"
      disabled={isLoading}
    >
      {Object.keys(availableModels)
        .filter((key) => {
          const model = availableModels[key];
          // Temporarily filter out shape models
          return model.category !== "shape";
        })
        .map((key) => {
          const model = availableModels[key];
          return (
            <option
              key={key}
              value={key}
              disabled={!model.available}
            >
              {model.name} {!model.available ? "(Không khả dụng)" : ""}
            </option>
          );
        })}
    </select>
  </div>
);

ModelSelector.propTypes = {
  modelType: PropTypes.string.isRequired,
  availableModels: PropTypes.object.isRequired,
  isLoading: PropTypes.bool,
  onChange: PropTypes.func.isRequired,
};

ModelSelector.defaultProps = {
  isLoading: false,
};

export default ModelSelector;


