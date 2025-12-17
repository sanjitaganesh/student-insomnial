# How Much Do I Study and How Much Do I Sleep?

This project analyzes whether students can compensate for lack of sleep with increased study hours, or if sleep deprivation negatively impacts academic performance.

Using the *Student Insomnia and Educational Outcomes* dataset, a regression model is built to predict academic performance based on sleep-related factors.

---

## Key Idea

Survey responses are **ordinal**, not numerical.  
Instead of forcing classification, the labels are converted into ordered values and treated as an **ordinal regression** problem.

Example:
Excellent = 4
Good = 3
Average = 2
Below Average = 1
Poor = 0

---

## Features Used

- Average sleep hours per day  
- Daytime fatigue frequency  
- Academic stress level  

Target:
- Overall academic performance

---

## Methodology

- Ordinal encoding of survey responses  
- Exploratory Data Analysis to observe trends  
- Linear Regression model  
- Evaluation using R² and Mean Absolute Error (MAE)

---

## Results
R² Score: 0.132
MAE: 0.617
Sleep and stress have a measurable effect, but they do **not fully explain** academic performance — suggesting all-nighters are neither a miracle nor the sole deciding factor.

---

## Conclusion

More study hours cannot reliably compensate for chronic sleep deprivation.  
Sleep quality and stress management play a meaningful but limited role in academic outcomes.
