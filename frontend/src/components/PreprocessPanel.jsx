import PropTypes from "prop-types";

const PreprocessPanel = ({ steps }) => {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="drawing-page__right">
      <h2 className="preprocess-title">Các bước tiền xử lý ảnh</h2>
      <div className="preprocess-steps">
        {steps.map((step, index) => (
          <div key={index} className="preprocess-step">
            <div className="preprocess-step__label">
              {index + 1}. {step.name}
            </div>
            <div className="preprocess-step__image-wrapper">
              <img
                src={step.image_base64}
                alt={step.name}
                className="preprocess-step__image"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

PreprocessPanel.propTypes = {
  steps: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      image_base64: PropTypes.string.isRequired,
    }),
  ),
};

PreprocessPanel.defaultProps = {
  steps: [],
};

export default PreprocessPanel;


