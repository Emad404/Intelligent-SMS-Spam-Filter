# Intelligent SMS Spam Filter

## Project Overview
This project is an **Intelligent Spam Filter** designed to classify SMS messages as either **spam** or **ham** (not spam). The system explores the implementation and effectiveness of three distinct machine learning techniques—**Decision Tree**, **Naive Bayes**, and **K-Nearest Neighbors (KNN)**—to detect patterns in text messages based on commonly used spam keywords.

## Implemented Techniques

### 1. K-Nearest Neighbors (KNN)
- **Methodology**: An instance-based learning approach that classifies new messages by comparing their feature vectors with the training dataset using **Euclidean distance**.
- **Preprocessing**: Includes lowercasing, punctuation removal, and stopword filtering to clean raw SMS content.
- **Performance**: Achieved an accuracy of **91.46%** on a test set of 1,394 messages.

### 2. Naive Bayes
- **Methodology**: A probabilistic classifier based on the Multinomial Naive Bayes model, assuming conditional independence of word tokens.
- **Performance**: This statistical approach adapts robustly to fresh data, achieving an accuracy of **98.42%** on the experimental dataset.

### 3. Decision Tree
- **Methodology**: A rule-based classifier implemented using the ID3 algorithm. It constructs a recursive tree by choosing keywords that maximize **Information Gain**.
- **Performance**: Provides high interpretability with a transparent logic for spam classification, achieving **89.81%** accuracy.

## Comparative Summary
| Feature | Naive Bayes | Decision Tree | K-Nearest Neighbors |
| :--- | :--- | :--- | :--- |
| **Learning Type** | Probabilistic | Rule-based | Instance-based |
| **Accuracy** | 98.42% | 89.81% | 91.46% |
| **Interpretability** | High | Very High | Low |
| **Training Requirement** | Fast | Moderate | No training required |

## Technical Stack
- **Language**: Java SE 17.
- **Tools**: Standard `java.util` and `java.io` packages for core logic; no external machine learning libraries were used for the primary implementations.
- **Data Processing**: Initial data cleaning and preparation were supported by Python and Jupyter Notebooks.

## Getting Started
### Prerequisites
- Java Development Kit (JDK) 17 or higher.

### Usage
1. Clone the repository to your local machine.
2. Navigate to the `Source_Code` directory.
3. Choose the specific algorithm subdirectory (KNN, Naive_Bayes, or Decision_Tree) to explore the implementation.
4. Run the main Java classes to evaluate the classifier against the provided SMS dataset.
