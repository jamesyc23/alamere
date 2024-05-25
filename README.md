# Alamere: Confidence-based cascade routing for LLMs
_This project was completed by Alexander Wu and James Chen at the 2024 RunPod Open-Source LLM Hackathon, and was awarded the Best Use of RunPod Serverless award._

__Motivation:__ We saw that there was a notable tradeoff between the quality of a given model's responses and the computational cost of inference for that model
(which is ~linear in number of parameters). As a concrete example, we found that a large model like Llama3 70B had 93% accuracy on GSM8K 5-shot CoT, but cost 80c/MTok
(80 cents per million tokens), whereas a small model like Llama3 8B had 75% accuracy under the same setup but cost 7c/MTok. In this project we explored how one might get
"the best of both worlds" -- that is, accuracy close to that of a large model at close to the cost of the small model.

(Image)

_We primarily experimented using Llama3 8B and Llama3 70B as the small model and large model, respectively, with the task being GSM8K 5-shot CoT._

__Summary__:
We implemented a confidence-threshold-based cascade routing system. For a given user request, we first pass it to the small model and compute a confidence 
score between 0 and 1 for the model's response. Then we check to see if the confidence is greater than a predetermined threshold (e.g. 80%):
* If yes, we return the answer to the user;
* If no, we pass the request to the large model and return the large model's response to the user instead.

Somewhat surprisingly, we found that a linear transformation of the log probability of the model's full response was a good predictor of a model's accuracy, achieving 90.6% r<sup>2</sup>
relative to a mean accuracy predictor with bin size 100.

The final implementation is as follows:
1. User inputs a collection of representative problems (we use these to fit the linear model coefficients and also to compute what the expected accuracy and cost are for a given threshold).
2. User determines the desired confidence threshold and deploys a model serving endpoint.

(Image)

See the above results / play around with the data for yourself in confidence.ipynb!

## Appendix

[__Hackathon Demo Video__](https://www.loom.com/share/0670862f1d2e4a14af76b129e53f0537)
[__Hackathon Presentation Slides__](https://docs.google.com/presentation/d/17zzomxlxOBamHTs4gzot2qo9yquAVc8Y2N0vuzEe-AE/edit?usp=sharing)
