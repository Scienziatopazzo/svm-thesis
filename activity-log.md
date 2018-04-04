# SVMRegression - Research Activity Log

## 4th April 2018
Started implementing scaling and doing outlier detection only on the training split of the data. TODO: outlier detection, proper pipeline.

## 3rd April 2018
* Experimented with polynomial kernel of deg>3, some cases make the computation too long, hanging the grid search.
* Experimented with scalers: `RobustScalers` retains outliers, hanging the grid search. `QuantileTranformer` obtains the same scores as the usual `StandardScaler`.
* Added the possibility of doing outlier detection and filtering on the dataset in `dogs_2006_2016.py`. Using this option seems to improve scores.

## 2nd April 2018
* Removed deprecated `set_value` calls in `dogs_2006_2016.py` and cleaned up the code.
* Cleaned up Training examples in notebooks, added a comparison between `GridSearchCV` and Repeated Houldout in the `Repeated Holdout` notebook.
* Modified thesis structure.

## 29th March 2018
Defined a first draft of the thesis structure.

## 25th March 2018
Implemented kernels in `svr.py` and nonlinear tests. Custom SVR consistently obtains the same scores of the Sklearn SVR in both linear and nonlinear tests.

**Note:** with the polynomial kernel, using negative values for `coef0` returns a Gurobi error. These values were removed from the holdout search, since they were seemingly never chosen anyways. Even though not specified in the sklearn documentation, a brief research seems to point at negative `coef0` values being unallowed in general. Strangely, Sklearn doesn't misbehave when fed those values.

## 23rd March 2018
Fixed a typo in `svr.py`, the SVR now performs very well.

## 22nd March 2018
Implemented a first version of a custom SVM Regression with Gurobi in `svr.py`. Did first tests on new notebook `Custom SVR Tests`, the SVM seems to perform poorly. Need to determine causes.

## 20th March 2018
Completed corrections of thesis draft.

## 20th November 2017
Continued work on thesis draft, reached "Censoring policies" in the "Preprocessing and exploration" section.

## 18th November 2017
Started working on first draft of thesis material in latex.

## 12th November 2017
Set up *Gurobi* and `gurobipy` and started experimenting with the solver, with the eventual goal of developing a modifiable SVM module and implementing the changes useful for dealing with censored data as described in *Van Belle et al*.

## 9th Nobember 2017
* Read *Van Belle et al, Support vector methods for survival analysis: a comparison between ranking and regression approaches*.
* Did 10 trials of holdout using both censoring policies in the `Repeated Holdout` notebook, then tested SVRs with the obtained hyperparameters. All with negative results.

## 6th November 2017
* Created `training` script with an implementation of *Holdout grid search*, together with the `Repeated Holdout` notebook. First experiments yielded negative results.

## 5th November 2017
* Updated `dogs_2006_2016.py`, removing uses of deprecated pandas method *set_value*.

## 29th October 2017
* Added `Preprocessing` check for *Age*. When computing age as *(death-birth)* almost all values were erroneous. When doing it as *(firstvisit-birth)* (age at the first visit) most values were correct but there were still 50 errors. The origin of the errors is unclear, as many erroneous values appear to have no relation with the provided dates.
* Created notebook `Feature engineering`:
  - Added new feature *Therapy to visit*, a time delta in days from the beginning of the therapy to the first visit. (By adding this measure to *Survival time* it is possible to obtain a survival time from the beginning of the therapy, thus fulfilling the objective of 21/10/17).
  - Tried out removal of highly correlated features in various combinations.
  - Explored the relationship between *IP Gravity* and *Vrig Tric* when both values are different from 0. IP Gravity is just a discretization of Vrig Tric, so there's no point in using both.
* Reviewed k-fold cross-validation. Started using more folds as using 2 was clearly bad practice. Benefits of using holdout over it (as instructed) are unclear.

## 28th October 2017
* Introduced filling of NA values in notebook `NA fill`, using different policies:
  - *mean* -  Using the mean of the feature.
  - *normal* - Generating values from a normal distribution modeled after the feature (checked approximate normality with a qqplot beforehand).
  - **TODO:** Using values predicted from a model (eg. linear regression) trained with correlated features.
* Introduced primitive methods for dealing with the right-censoring in *Survival time* in notebook `Censoring`:
  - *drop* - just drop all data for which *Dead*==0.
  - *max* - when *Dead*==0, replace all *Survival time* values with the maximum Survival time of data for which *Dead*==1.

Combining the *normal* NA policy with *drop* censoring policy an SVR with grid search was able to obtain a slightly positive score, although not stable.

## 25th October 2017
* New information about the dataset:
  - Obtained description of *isachc* scale. *C* means that the dog was only classified with the newer scale, so it's equivalent to *NA*. As suggested, the feature was deemed probably irrelevant.
  - Resolved issue posed by the fact that all dogs had a death date, even alive ones. The death date is in fact right-censored for dogs that were last recorded as alive.

## 23rd October 2017
Added attribute *fixErrors* to *load_df_dogs_2016* in `dogs_2006_2016.py`, for fixing the one wrong *Survival time* value in the dataset.

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
