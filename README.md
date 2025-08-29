# 🤖 AI Model Latency Profiler

A Flask-based single page application for measuring and comparing inference times across different machine learning models. This tool helps developers understand the performance trade-offs between different model architectures and complexities.

## 🎯 Purpose

- **Measure inference latency** of different ML models
- **Compare performance** across model types
- **Visualize results** with interactive charts
- **Learn trade-offs** between model complexity and speed

## 🚀 Features

### Models Included
- **Logistic Regression**: Linear model - fast inference, limited complexity
- **Random Forest**: Ensemble model - moderate speed, good accuracy
- **Linear SVM**: Moderate speed, good for linear separability
- **RBF SVM**: Slower, handles non-linear patterns
- **Small Neural Network**: Moderate speed, some non-linearity
- **Large Neural Network**: Slower, high complexity

### Key Features
- **Interactive Model Selection**: Choose which models to compare
- **Configurable Testing**: Adjust number of inference runs
- **Real-time Charts**: Visualize latency comparisons and distributions
- **Statistical Analysis**: Mean, standard deviation, min/max times
- **Modern UI**: Beautiful, responsive interface

## 🛠️ Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python app.py
   ```
4. **Open your browser** and go to `http://localhost:5001`

## 📖 How to Use

### Step 1: Train Models
- Click the "🚀 Train All Models" button
- Wait for all models to be trained on synthetic classification data
- This may take a few moments

### Step 2: Select Models
- Choose which models you want to compare
- Use "Select All" or "Deselect All" for convenience
- Each model shows its description and training time

### Step 3: Profile Latency
- Set the number of inference runs (default: 100)
- Click "⚡ Start Profiling" to measure inference times
- View results with interactive charts

### Step 4: Analyze Results
- **Statistics Cards**: See mean inference times and standard deviations
- **Bar Chart**: Compare mean latency across models
- **Line Chart**: View latency distribution over multiple runs

## 📊 Understanding the Results

### Latency Metrics
- **Mean Time**: Average inference time across all runs
- **Standard Deviation**: Consistency of inference times
- **Min/Max**: Best and worst case performance

### Model Performance Insights
- **Linear models** (Logistic Regression) are typically fastest
- **Ensemble methods** (Random Forest) offer good speed-accuracy balance
- **Neural networks** provide flexibility but at higher computational cost
- **SVM with RBF kernel** is slowest due to kernel computations

## 🎨 Technical Details

### Backend (Flask)
- **Model Training**: Uses scikit-learn for various ML models
- **Latency Measurement**: Precise timing with warm-up runs
- **API Endpoints**: RESTful endpoints for training and profiling
- **Threading**: Non-blocking model training

### Frontend (HTML/CSS/JavaScript)
- **Responsive Design**: Works on desktop and mobile
- **Chart.js Integration**: Interactive data visualization
- **Modern UI**: Gradient backgrounds, smooth animations
- **Real-time Updates**: Dynamic content loading

### Data
- **Synthetic Dataset**: 1000 samples, 20 features, binary classification
- **Standardized Features**: Preprocessed for fair comparison
- **Consistent Testing**: Same data used for all models

## 🔧 Customization

### Adding New Models
1. Add model configuration in `train_models()` function
2. Include model class, description, and color
3. Models will automatically appear in the UI

### Modifying Test Parameters
- Change `num_runs` in the UI for different precision
- Modify `make_classification` parameters for different data
- Adjust model hyperparameters in the configuration

## 🐛 Troubleshooting

### Common Issues
- **Models not training**: Check scikit-learn installation
- **Charts not loading**: Ensure internet connection for Chart.js CDN
- **Slow performance**: Reduce number of runs or use fewer models

### Performance Tips
- Use fewer models for faster comparison
- Reduce number of runs for quick testing
- Close other applications to free up CPU

## 📈 Learning Outcomes

This tool demonstrates:
- **Performance vs. Complexity trade-offs**
- **Importance of model selection for production**
- **Impact of model architecture on inference speed**
- **Statistical analysis of ML model performance**

Perfect for developers learning about ML model deployment and optimization!
