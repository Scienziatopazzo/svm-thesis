# SVMRegression - Research Activity Log

## 23rd October 2017
* Added attribute *fixErrors* to *load_df_dogs_2016* in `dogs_2006_2016.py`, for fixing the one wrong *Survival time* value in the dataset.

## 22nd October 2017
* Conducted first exploratory analysis of the dogs dataset in `Dataset Analysis` notebook.
  - The only visible correlations with *Survival time* detected were with *Age* and *Asx/ao* (and in minor proportion *FE %*)
* Tried out various SV regressions in `Dataset Analysis` notebook, all with awful scores.
* *"Solved"* the apparent issue exposed on 21/10/17: some date entries were automatically transcribed by excel into the American format, but they are clearly recognizable by `converters.py` as such, so data consistency is maintained.

## 21st October 2017
* Finished first draft of `Preprocessing` notebook. Notes:
  - All detected date inconsistencies are related to therapies started before the first visit. Could possibly refer to therapies started with other clinics. Therefore potentially not an error, although the current survival time measure does not take this into account and it may be useful to try computing one that does.
  - A single error of 365 days in survival time was detected. Survival time values were written by hand, so the detected error was probably the result of misreading the year of an entry.
  ~~**Note:** this means that if the entirety of the "Date of death" column uses American date formats, then whoever transcribed "Survival time" read the wrong value for every date that can be also read as European. (Our program only interprets dates with the American format if they can't be understood as European)~~ (not an issue, look at 22/10/17)
  - No inconsistencies found for *cardiac arrest death -> death* and the *Therapy category* value
* Tried out Sklearn SVR module with matplotlib in `Sklearn` notebook.
* Created `Dataset Analysis` notebook.

## 20th October 2017
Started working on preprocessing the dogs dataset.

## 10th-17th October 2017
Read *Smola and Schölkopf, A Tutorial on Support Vector Regression*.

## August-September 2017
Read *Burges, A Tutorial on Support Vector Machines for Pattern
Recognition*.
