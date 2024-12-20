# CSI4900: Textual Emotion-Cause Pair Extraction in Conversations

##### Ayoub
##### Arthur
##### Mershab

#### Supervising Professor: Diana Inkpen



# Table of Contents

1. **Introduction**  
   - 1.1 Background  
   - 1.2 Motivation  
   - 1.3 Objectives  

2. **Related Work**  
   - 2.1 Emotion Recognition in Text  
   - 2.2 Cause-Pair Extraction Techniques  
   - 2.3 Applications in Conversational AI  

3. **Methodology**  
   - 3.1 Fine-Tuning a Model for Emotion Classification  
     - 3.1.1 Dataset Selection and Preprocessing  
     - 3.1.2 Model Architecture  
     - 3.1.3 Training Process  
   - 3.2 Building the Emotion-Cause Pair Extraction Model  
     - 3.2.1 Leveraging Hugging Face Transformers  
     - 3.2.2 Annotation and Cause-Pair Representation  
     - 3.2.3 Model Design and Implementation  
   - 3.3 Development of a Demonstration Application  
     - 3.3.1 Demo Design Goals  
     - 3.3.2 Integration of Emotion and Cause-Pair Models  
     - 3.3.3 User Interaction and Visualization  

4. **Results and Analysis**  
   - 4.1 Emotion Classification Performance  
   - 4.2 Cause-Pair Extraction Accuracy  
   - 4.3 Evaluation Metrics and Comparative Analysis  

5. **Challenges and Solutions**  
   - 5.1 Data Quality and Annotation Issues  
   - 5.2 Model Limitations  
   - 5.3 Mitigation Strategies  

6. **Conclusion and Future Work**  
   - 6.1 Key Takeaways  
   - 6.2 Potential Extensions  
   - 6.3 Closing Remarks  

7. **References**  

8. **Appendices**  
   - 8.1 Code Snippets and Configurations  
   - 8.2 Additional Experiment Details  
   - 8.3 Supplementary Materials  

---

# 1. Introduction

## 1.1 Background  
The ability to recognize emotions and their underlying causes in textual data has become a pivotal aspect of conversational AI systems. With the proliferation of virtual assistants and social media platforms, understanding emotional context is critical for enhancing user interactions and providing meaningful responses. Emotion Cause-Pair Extraction (ECPE) is a challenging task that aims to identify not only the emotions expressed in text but also the specific textual causes of those emotions.

## 1.2 Motivation  
Despite significant advancements in emotion recognition, the explicit linking of emotions to their causes remains an underexplored area. This gap is particularly evident in conversational settings, where contextual nuances play a vital role. By addressing this challenge, we can improve the interpretability of AI systems, making them more effective in applications such as mental health support, customer service, and sentiment analysis.

## 1.3 Objectives  
The primary objectives of this report are:  
1. To fine-tune a competitive model for emotion classification.  
2. To develop a robust system for extracting emotion-cause pairs using the Hugging Face Transformers library.  ##TODO: Arthur fix this if necessary
3. To build a demonstration application that showcases the integration of these models in real-world scenarios.  

This report outlines the methodology, results, and challenges encountered during the development of the ECPE system, providing insights into its potential applications and limitations.

---

# 2. Related Work

## 2.1 Emotion Recognition in Text  
_placeholder_

## 2.2 Cause-Pair Extraction Techniques  
_placeholder_

## 2.3 Applications in Conversational AI  
_placeholder_

---

# 3. Methodology

## 3.1 Fine-Tuning a Model for Emotion Classification  

### 3.1.1 Dataset Selection and Preprocessing  
_placeholder_

### 3.1.2 Model Architecture  
_placeholder_

### 3.1.3 Training Process  
_placeholder_

## 3.2 Building the Emotion-Cause Pair Extraction Model  

### 3.2.1 Leveraging Hugging Face Transformers  
_placeholder_

### 3.2.2 Annotation and Cause-Pair Representation  
_placeholder_

### 3.2.3 Model Design and Implementation  
_placeholder_

## 3.3 Development of a Demonstration Application  

### 3.3.1 Demo Design Goals  

1. **Simplicity**  
   The demo aims to provide a straightforward and intuitive user interface that makes it easy for users to interact with the models. By focusing on a conversational paradigm, users can naturally input text and observe the corresponding emotional insights and cause-pair relationships.  

2. **Reproducibility**  
   To ensure that the demo can be easily reproduced and deployed in different environments, it is packaged as a Helm chart for Kubernetes. This approach simplifies the deployment process, allowing users to spin up the application in a cloud or local environment with minimal setup effort.  

3. **Showcasing the Emotion-Cause Pair Extraction**  
   The demo highlights the unique capability of identifying emotion-cause pairs within text. By leveraging a conversational interface, the system dynamically demonstrates how emotions are detected and linked to their causes, providing an engaging and educational experience.  

### 3.3.2 Integration of Emotion and Cause-Pair Models  
_placeholder_

### 3.3.3 User Interaction and Visualization  
_placeholder_

---

# 4. Results and Analysis

## 4.1 Emotion Classification Performance  
_placeholder_

## 4.2 Cause-Pair Extraction Accuracy  
_placeholder_

## 4.3 Evaluation Metrics and Comparative Analysis  
_placeholder_

---

# 5. Challenges and Solutions

## 5.1 Data Quality and Annotation Issues  
_placeholder_

## 5.2 Model Limitations  
_placeholder_

## 5.3 Mitigation Strategies  
_placeholder_

---

# 6. Conclusion and Future Work

## 6.1 Key Takeaways  
_placeholder_

## 6.2 Potential Extensions  
_placeholder_

## 6.3 Closing Remarks  
_placeholder_

---

# 7. References  
_placeholder_

---

# 8. Appendices

## 8.1 Code Snippets and Configurations  
_placeholder_

## 8.2 Additional Experiment Details  
_placeholder_

## 8.3 Supplementary Materials  
_placeholder_
