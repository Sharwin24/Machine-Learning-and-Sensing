# Feature Extraction
Each feature extraction method had different feature vectors that they utilized to describe the audio file:
\begin{table}[H]
    \centering
    \begin{tabular}{c|c|c}
      FFT & MFCC & RMS \\
      \hline
      $\mu$, $\sigma$, $E_{total}$, $E_{low}$, $E_{mid}$, $E_{high}$ & $\mu$, $\sigma$, $\Delta\mu$, $\Delta\sigma$, $\Delta_2\mu$, $\Delta_2\sigma$ & $\mu$, $\sigma$,$\max(RMS)$, $\min(RMS)$, $\mu_{zcr}$, $\sigma_{zcr}$
    \end{tabular}
    \caption{Feature Vectors for each extraction method}
    \label{tab:feature-table}
\end{table}
The features are normalized in all 3 vectors so they can be equally weighted and compared against each other. The window size or length of the audio file was also configured in the preprocessing step. There was no noise reduction explicitly done but the input audio data was relatively un-noisy due to the recorder app smoothing out the audio a bit.
# Model Training & Comparison
Model accuracies on the test set using the full audio file.
\begin{itemize}
  \item Random Forest: 0.68
  \item Logistic Regression: 0.82
  \item SVM: 0.78
  \item MLP: 0.88
\end{itemize}
\begin{table}[H]
    \centering
    \begin{tabular}{c|c|c}
      Model & Test Accuracy & Test Set
      \hline
      Random Forest & 0.68 & Large \\
      Logistic Regression & 0.82 & Large \\
      SVM & 0.78 & Large \\
      MLP & 0.88 & Large \\
      \hline   
      Random Forest & 0.74 & Small \\
      Logistic Regression & 0.70 & Small \\
      SVM & 0.70 & Small \\
      MLP & 0.78 & Small \\   
    \end{tabular}
    \caption{Feature Vectors for each extraction method}
    \label{tab:feature-table}
\end{table}
# Time Window Analysis
# Hyperparameter Optimization
Best parameters found:  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 10}
Best cross-validation score:  0.9800000000000001
Test set score:  0.68


# Model Evaluation on Holdout Data
The model confused most activities with laughing since it contained similar features to all other activities: it was cyclic in nature, had relatively similar volume levels.