# SVMRegression - Research Activity Log

## August-September 2017
Read *Burges, A Tutorial on Support Vector Machines for Pattern
Recognition*.

## 10th-17th October 2017
Read *Smola and Schölkopf, A Tutorial on Support Vector Regression*.

## 20th October 2017
Started working on preprocessing the dogs dataset.

## 21st October 2017
* Finished first draft of `Preprocessing` notebook. Notes:
  - All detected date inconsistencies are related to therapies started before the first visit. Could possibly refer to therapies started with other clinics. Therefore potentially not an error, although the current survival time measure does not take this into account and it may be useful to try computing one that does.
  - A single error of 365 days in survival time was detected. Survival time values were written by hand, so the detected error was probably the result of misreading the year of an entry.
  **Note:** this means that if the entirety of the "Date of death" column uses American date formats, then whoever transcribed "Survival time" read the wrong value for every date that can be also read as European. (Our program only interprets dates with the American format if they can't be understood as European)
  - No inconsistencies found for *cardiac arrest death -> death* and the *Therapy category* value
* Tried out Sklearn SVR module with matplotlib in `Sklearn` notebook.
* Created `Dataset Analysis` notebook
