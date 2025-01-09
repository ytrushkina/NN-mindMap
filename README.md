# A Mindmap for an NN Project

## Background: NN as a Function

### Which Function?
- Can create an infinitely flexible function from this one tiny thing *F.relu(m*x+b)
- Well, many of these things plus a simple non-linearity to separate them:
  ```python
  def rectified_linear(m, b, x):  # relu = rectified linear unit
      y = m * x + b  # Matrix multiplication, which is just multiplying things together and then adding them up
      return torch.clip(y, 0.)  # or F.relu(x) in PyTorch. I.e. the function max(x, 0), which simply replaces all negative numbers with zero
- different architectures: CNNs, RNNs, Transformers

## Pipeline:
- architecture (layers and forward pass function, i.e. how the layers connect)
- loss function
- optimizer
- training scheduler
- train for a #epochs (or use others stops)
- check the performance (e.g. visualize the loss curve)
- optimize

### Training = fitting the function to the data
- weights initialization: random (common in NNs), Xavier/Glorot (common in DL with sigmoid or tanh activations), He (common in DL with ReLu), pre-trained (transfer learning)
- prediction
- Learning Rate Schedulers: StepLR, ExponentialLR, CosineAnnealingLR
- loss: Mean Squared Error (MSE, related to L2 norm) for regression, Cross-Entropy Loss for classification, Contrastive loss function (multimodal search)
	- metrics to measure distance: Mean Absolute Difference (aka L1 norm), RMSE (Root Mean Square Error, aka L2 norm)
      - cosine distances depend on angles between embeddings
      - eucledian distance minimizes proximity between similar points
- optimization: updating weights to decrease loss with gradients
	- optimizers: Stochastic Gradient Descent (SGD), Adam, RMSprop	
	- bachpropagation: the process of calculating the derivative (gradient) of each layer
- how fast we update weights = learning rate => good learning rate = training performance optimisation
	- Learning Rate Schedulers: StepLR, ExponentialLR, CosineAnnealingLR
- when to stop to avoid under-/overfitting
  - early stopping (performance stops improving), Convergence Criteria (change in loss function < threshold), max epochs

### Data prep
- Normalization: if some inputs are way bigger than others, they will dominate the decision => normalization (make all inputs between 0 and 1 OR take logs)
- Data Augmentation: Techniques like rotation, flipping, cropping to increase dataset size and variability
- Handling Missing Data: Imputation, removal, or using models that can handle missing values

### Data output
- pydantic to ensure correct structured output

### Optimizing training 
- Learning Rate Schedulers: StepLR, ExponentialLR, CosineAnnealingLR
- Regularization: Techniques to prevent overfitting
		- L1/L2 Regularization
		- Dropout
		- Batch Normalization
- Early Stopping: Stop training when performance on a validation set starts to degrade
- Hyperparameter Tuning: Grid search, random search, Bayesian optimization

### Optimizing inference:
•	[NLP with Transformers](https://github.com/nlp-with-transformers/notebooks)
 - Model Pruning: Removing less important weights to reduce model size
 - Quantization: Reducing the precision of the numbers used to represent the model parameters
 - Knowledge Distillation: Training a smaller model to mimic a larger model
 - Deployment: Techniques for deploying models efficiently (e.g., TensorRT, ONNX)
   
•	[Efficiently Serving LLMs](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)
   - Speeding up text generation with KV-caching
   - Batching (processing multiple inputs at once: but! trading throughput with latency = user has to wait for the batch)
   - continuous batching: constantly swap out requests from the batch that have completed generation for requests in the queue that are waiting to be processed
   - Quantization
   - low-rank adaptation (LoRa), multi-lora inference, LoRaX


### Advanced techniques
- Transfer Learning: Using pre-trained models and fine-tuning them on new tasks
- Ensemble Methods: Combining multiple models to improve performance
- Attention Mechanisms: Allowing the model to focus on important parts of the input
- Generative Models: GANs, VAEs for generating new data
