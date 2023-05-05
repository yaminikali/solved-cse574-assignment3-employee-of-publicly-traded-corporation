Download Link: https://assignmentchef.com/product/solved-cse574-assignment3-employee-of-publicly-traded-corporation
<br>
<h1>Role: Employee of Publicly Traded Corporation</h1>

In this assignment your group takes the role of machine learning engineers who have been tasked with designing a new system for use in U.S. court systems. You are given 3 different ML models and must apply 5 postprocessing techniques onto each of them. Finally, your group must determine a single model and technique to submit, which will then be measured against the rest of the class for a chance at extra credit.

<h2>1 – Problem Setup</h2>




In 2016 the independent non-profit news organization ProPublica released a report evaluating Northpointe’s COMPAS system, which is an algorithm widely used across the country for considerations in pretrial detention and sentence determination.

<em>Propublica Story</em> – <a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">https://www.propublica.org/article/machine</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">–</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">bias</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">–</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">risk</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">–</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">assessments</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">–</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">in</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">–</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">criminal</a><a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">sentencing</a>

<em>Northpointe’s Response –</em> <a href="https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html">https://www.documentcloud.org/documents/2998391</a><a href="https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html">–</a><a href="https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html">ProPublica</a> <u>        </u><a href="https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html">Commentary</a><a href="https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html">–</a><a href="https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html">Final</a><a href="https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html">–</a><a href="https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html">070616.html</a>

COMPAS evaluates criminals on over 100 factors and outputs a risk score which indicates how likely it is that someone will recidivate (go on to commit another crime in the future). These scores are then taken into consideration by judges when assigning sentences, determining bail/parole eligibility, etc. Critically, race is not one of the factors used in by COMPAS.

ProPublica reviewed the output of COMPAS on a dataset of over 7000 individuals from Broward County,

Florida. They found that the algorithm correctly predicted recidivism at similar rates for both white (59%) and black defendants (63%). However, when the algorithm was incorrect it tended to skew very differently for each of these groups. White defendants who re-offended within two years were mistakenly labelled low-risk almost twice as often as their black counterparts. Additionally, black defendants who did not recidivate were rated as high-risk at twice the rate of comparable white defendants.

Northpointe submitted a rebuttal of ProPublica’s evaluation, claiming that PP misrepresented certain statistical values. They assert that their model is entirely fair across racial lines when base rate recidivism levels are taken into account.

On the heels of these reports your company has decided to release a new system as a potential candidate for replacing COMPAS. 3 machine learning models have been designed by your development team and trained on the Broward County data. These include:

<ul>

 <li>A linear support vector regressor</li>

 <li>A feed forward neural network</li>

 <li>A naïve Bayes classifier</li>

</ul>




In addition, your research team has been scanning machine learning research papers and have determined 5 potential post-processing methods that enforce various constraints in attempts to reflect different measures of fairness. These include:

<ul>

 <li>Maximum profit / maximum accuracy</li>

 <li>Single threshold</li>

 <li>Predictive parity</li>

 <li>Demographic parity</li>

 <li>Equal Opportunity</li>

</ul>

As the lead development team of your company it falls upon your group to implement these 5 methods on each model. You must then determine which model/fairness combination to put forward as your finished product. Since this is an important social issue, you are certain that rival companies and socially inclined NGOs will also be releasing similar systems. You need to be sure to sufficiently justify all algorithmic/ethical considerations about your model so that it can withstand the scrutiny of the public eye and ultimately be chosen as the replacement for COMPAS.




1.1 – Your Role

You will approach this assignment as employees of a large publicly traded corporation. You have a fiduciary duty to your shareholders to deliver strong returns, which can only be done by consistently winning development contracts. To this end, financial considerations are very important, as you want your contracts to be more appealing than your competitors. Additionally, your life relies on the paycheck provided by this job, and making decisions that lose the company money could very quickly get you fired. However, since your corporation is publicly traded you are still subject to public scrutiny, and thus cannot completely ignore social concerns.







<h2>2 – Coding Tasks</h2>




2.1 – Models &amp; Data

As listed above, you are given 3 working machine learning models: a linear support vector regressor

(<em>Compas_SVM.py</em>), a feed-forward neural network (<em>Compas_NN.py</em>), and a naïve Bayes classifier (<em>Compas_Naive_Bayes.py</em>). Each of these models has a single function which loads the data, builds and runs the model, and outputs a set of predictions alongside the rest of the useful data needed for evaluation. The data consists of 4 files: <em>Compas_train_data.npy, Compas_train_labels.npy, </em>

<em>Compas_test_data.npy, </em>and <em>Compas_test_labels.npy</em>, all of which are contained within

<em>Broward_Data.zip. </em>All files provided to you should be extracted/downloaded into the same directory for them to run properly.

Running one of the model files will load the data, classify the data, separate all relevant predictions and labels into groups based on race, and call the <em>report_results </em>function from <em>Report_Results.py</em>. This function calls each of the 5 post-processing methods (which you must complete) and prints out some useful metrics that can be used to determine if your functions are working correctly. The <em>report</em>_<em>results </em>function will be used by the graders on these models in order to determine the accuracy of your final solution.

Additionally, you are provided with <em>utils.py</em>, a collection of functions used to gather various metrics from the classified data. You are welcome to write your own functions, but everything you need for completing the postprocessing methods is provided within this file.




2.2 – Postprocessing

This assignment has two main parts: implementation and evaluation. The implementation section requires you to finish methods for 5 different postprocessing techniques, which can be found in <em>Postprocessing.py</em>. The function headers have all been written for you, but your team must complete the function bodies. Each function takes <em>categorical_results </em>as an input. This is a dictionary – the keys are the 5 different racial groups, and the entry for each key is a list of (prediction, label) tuples for all people within that racial group.

Some functions also take <em>epsilon </em>as an additional parameter, which acts as a tolerance for enforcing some of the fairness constraints. The specific <em>epsilon </em>values you need to satisfy are included alongside each specific function. <strong>NOTE: <em>The listed epsilon values must be consistently upheld for the SVM and Naïve Bayes Classifier. The neural network may occasionally fail or provide trivial solutions using these values.</em></strong>

While COMPAS gives a recidivist rating of 1-10, with additional labels of low, medium, and high risk, the models you have been given for this assignment all output a list of real-valued predictions between [0, 1]. The postprocessing methods you must implement will determine threshold values for these predictions; anything above the value will be a 1, and anything below the value will be a 0. This will give a final result of a list of binary decisions. Some functions will require you to find different threshold values for each racial group.

These functions must return the two things: the classified data which has been thresholded to meet the requirements of the function, and the thresholds themselves. The output data should be in the same form as the input data. The thresholds should also be a dictionary with racial groups as the keys, but the entries will be the scalar threshold value for that racial group.

Definitions of necessary terms and a few basic pros/cons of each method are provided in the supplementary handout, <em>Machine_Learning_Fairness_Primer.pdf </em>




<em>Maximum Profit / Maximum Accuracy / Minimal Loss (10pts) </em>

The maximizing method produces the best possible results from the given classifier. “Best” is contextspecific; it can refer to metrics such as the highest accuracy, the maximum profit the minimal loss. This function requires finding thresholds for each racial group to maximize your chosen secondary optimization criteria (either cost or accuracy, see section 2.3).




<em>Single Threshold (10pts) </em>

Arguably the simplest of the thresholding methods, single threshold holds all groups to the same decision standard by using one threshold value for all decisions.




<em>Predictive Parity (10pts) </em>

A classifier that enforces predictive parity will have the same positive predictive value (PPV) for each group. According to Northpointe, the COMPAS system satisfies predictive parity across racial groups when base rates of recidivism are accounted for. Your method must enforce predictive parity within an epsilon tolerance of ±1%.




<em>Demographic Parity (10pts) </em>

Demographic parity means that each group has an equal proportion of members classified in each way. Since our system is binary, this means that each group must have the same percentage of people who are classified as recidivists. Your function must be able to enforce demographic parity within an epsilon tolerance of ±2%.




<em>Equal Opportunity (10pts) </em>

Equal opportunity refers to all groups having the same opportunity to achieve the “privileged” outcome. This terminology can be somewhat misleading, as in our case it refers to being labelled as a recidivist. Enforcing equal opportunity means that all groups will have the same true positive rate (TPR). Your functions must be able to enforce equal opportunity within an epsilon tolerance of ±1%.




To be clear, <strong>you do not need to implement each of these methods 3 times. </strong>If you correctly implement the functions in <em>Postprocessing.py</em> they should work for all 3 of the provided models.




2.3 – Secondary Optimization

Many of these constraints can be satisfied using many different threshold values. For example, a singlethreshold system can be achieved by choosing <em>any</em> number between 0 and 1 as a threshold. Of course, your group wants the very best results, and those can only be achieved by performing a secondary optimization.

Two metrics can be used for this secondary optimization: <strong>accuracy</strong> and <strong>financial cost</strong>. For each one of the 5 post-processing methods you must return a set of thresholds that both satisfy the necessary constraint and also return the best possible overall value for one of these two metrics. You are welcome to choose either one, but you must use the same secondary optimization metric for all 5 functions. You must list the metric you have chosen in the space provided at the top of <em>Postprocessing.py</em>.




<h2>3 – Evaluation</h2>




3.1 – Report Essentials

After you have implemented the 5 postprocessing methods, you must choose a single model and postprocessing method as your group’s market submission. More details about this aspect of the project are provided in Section 5.

As explained in the supplementary handout, there is no definitive answer to what constitutes fairness in machine learning. Therefore, you must provide a detailed report justifying the model you choose to submit. The report must be <strong>no longer than 2 pages</strong>. There is no minimum length, as long as you have fulfilled the criteria listed below. The beginning of this report <strong>MUST</strong> include your model choice, algorithm choice, secondary optimization criteria (cost or accuracy), the overall cost of your system’s choices to society, and your overall accuracy.

The following questions are necessary for the body of report:

<ul>

 <li>What is the motivation for creating a new model to replace COMPAS? What problem are you trying to address? (8pts)</li>

 <li>Who are the stakeholders in this situation? (8pts)</li>

 <li>What biases might exist in this situation? Are there biases present in the data? Are there biases present in the algorithms? (8pts)</li>

 <li>What is the impact of your proposed solution? (8pts)</li>

 <li>Why do you believe that your proposed solution a better choice than the alternatives? Are there any metrics (TPR, FPR, PPV, etc?) where your model shows significant disparity across racial lines? How do you justify this? (8pts)</li>

</ul>




Additionally, 5pts of the report will come from professionalism. You must reference at least one  outside source in support of your argument. This can be either one that we’ve provided, or one you’ve found on your own. <strong>Do not just rely on the pros/cons provided in the fairness primer.</strong>

Finally, the remaining 5 pts of the report will come from ethical consistency. Does your argument make sense, and do you show that you have a strong understanding of the case you’re presenting? No stance is inherently wrong, but you must be able to justify your position without contradicting yourself.




3.2 – Report Extra Credit

Answering these questions constitutes the bare minimum that you should provide. 5% extra credit will be provided to the group that provides the best justification, se we strongly suggest that you include additional information and references to papers that support your choice. We have provided an incomplete list of extra topics to potentially include. You are also welcome to provide any other information or arguments that you consider relevant.

<ul>

 <li>How do you justify valuing one metric over the other as constituting “fairness”?</li>

 <li>What assumptions are made in the way we have presented the assignment? Are certain answers presupposed by the way we have phrased the questions?</li>

 <li>In what ways do these simplifications not accurately reflect the real world?</li>

 <li>How do uncertainty and risk tolerance factor into your decision?</li>

 <li>To what extent should base rates of criminality / recidivism among different groups be factored into your decision?</li>

 <li>The tools we provide can split the predictions into different protected categories, such as by age or gender. What disparities arise in these groups? How do these disparities compare to those shown when the predictions are split by race?</li>

</ul>




<h2>4 – Financials</h2>




A system such as this has far-reaching social implications for the real world. One such consequence to be considered is the monetary cost of the decisions provided by your model. Each type of classification carries a unique cost or benefit to society as a whole. These values are detailed below:




True Positives:

-$60,076 per person per year

True positives reflect people who were correctly predicted to recidivate. The cost for this type of prediction is the cost of keeping one inmate housed in jail for one year in New York State. This value includes the cost of living and upkeep for an inmate and the fractional salary and benefits of the corrections officers needed to oversee the inmates.

Source: <a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">https://storage.googleapis.com/vera</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">web</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">assets/downloads/Publications/price</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">of</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">prisons</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">what</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">incarceration</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">costs</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">taxpayers/legacy_downloads/price</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">of</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">prisons</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">updated</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">version</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">–</a><a href="https://storage.googleapis.com/vera-web-assets/downloads/Publications/price-of-prisons-what-incarceration-costs-taxpayers/legacy_downloads/price-of-prisons-updated-version-021914.pdf">021914.pdf</a>




True Negatives:

+$23,088 per person per year

True negatives reflect people who were correctly predicted not to re-offend. This is actually a positive value, and is determined by the annual pre-tax earnings of a full-time worker making minimum wage (2019) in NYS. While this value is somewhat simplified and doesn’t necessarily represent all of the social value created by a functioning member of society, it is often used as the baseline in criminology research.

Source: <a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">https://www.ny.gov/new</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">–</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">york</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">–</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">states</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">–</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">minimum</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">–</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">wage/new</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">–</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">york</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">–</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">states</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">–</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">minimum</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">–</a><a href="https://www.ny.gov/new-york-states-minimum-wage/new-york-states-minimum-wage">wage</a>




False Positives:

-$110,076 per person per year

False positives represent people who were incorrectly predicted to recidivate. While this prediction may not directly lead to a jail sentence, there are many U.S. districts that rely very heavily on systems such as COMPAS. A false positive could thus potentially represent a wrongfully imprisoned person, and this value reflects the cost of housing one person in jail plus the compensation cost for wrongful imprisonment. Many states have laws and precedents that set an expected value to be collected by a lawsuit brought forth from wrongful imprisonment. NYS doesn’t actually have a limit on this amount, but the most common value from other states is $50,000 per year.

Source: <a href="https://www.innocenceproject.org/wp-content/uploads/2017/09/Adeles_Compensation-Chart_Version-2017.pdf">https://www.innocenceproject.org/wp</a><a href="https://www.innocenceproject.org/wp-content/uploads/2017/09/Adeles_Compensation-Chart_Version-2017.pdf">–</a><a href="https://www.innocenceproject.org/wp-content/uploads/2017/09/Adeles_Compensation-Chart_Version-2017.pdf">content/uploads/2017/09/Adeles_Compensation</a><a href="https://www.innocenceproject.org/wp-content/uploads/2017/09/Adeles_Compensation-Chart_Version-2017.pdf">Chart_Version</a><a href="https://www.innocenceproject.org/wp-content/uploads/2017/09/Adeles_Compensation-Chart_Version-2017.pdf">–</a><a href="https://www.innocenceproject.org/wp-content/uploads/2017/09/Adeles_Compensation-Chart_Version-2017.pdf">2017.pdf</a>




False Negatives:

-$202,330 per person per year

False negatives represent people who were incorrectly predicted to not recidivate and then went on to commit another crime. Our value represents the average annual cost of a crime on society, a topic which is understandably highly contentious in legal literature. Different crimes obviously have different associated values, but since this system is being deployed in the real world it is a reasonable assumption that we won’t know what type of crime was committed beforehand.

A baseline for this value was obtained from the source listed below, but additional factors were included to reflect the nature of this assignment<strong>.</strong> We adjusted their calculations for inflation, and slightly skewed them upwards to represent the fact that we are calculating crime cost per year. Non-violent crimes are significantly more likely to happen, meaning that there is a much higher chance of a single offender committing multiple crimes per year before getting arrested.

Source: <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2835847/pdf/nihms170575.pdf">https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2835847/pdf/nihms170575.pdf</a>




These values will be applied to the results of your model to determine the total cost to society. This metric will be considered in the market evaluation, described in section 5.




<h2>5 – Market Evaluation Competition</h2>




Since fairness is a contested subject within machine learning, we have no authority to proclaim that one of these solutions is objectively better than the others. Oftentimes in society the moral choice at any given moment is determined by the popular voice, for better or worse. While it is up to you to justify why your solution is better than the alternatives, the decisions of the whole class will determine what constitutes the “best” fairness metric in the context of a market game termed <em>Vox Populi</em>.

The basic idea is that a method will only qualify as “fair” if it obtains a certain percentage of approval from the total class. Methods falling below this threshold percentage will be considered “unfair” and subsequently disregarded.

In the first round of the competition the popularity of each postprocessing algorithm will be calculated as a percentage of the total submissions. Any function used by 20% or more submissions will be considered a de facto standard for fairness. The remaining techniques will be deemed unfair by our mock society, and any submission that uses them will be not be considered for the next round of the competition.

The second round will look at the secondary optimization metrics. All fair models will be checked to see whether accuracy or profit was considered to be the more important secondary optimization metric overall. This chosen metric will then be used to determine the best performing model of the class.

For this competition you are allowed to freely change the model you decided to use. This includes adjusting any hyper-parameters, seeds, kernels, or network structures. If you really want to you can build your own model, as long as it matches the criteria listed below. You are also free to change the data metrics used by your model, which is determined by the metric list provided to the <em>preprocess</em> function. However, <strong>you must use the same secondary optimization method that you declared in your report</strong>. This model will be tested on the standardized dataset that has been provided to you, and we must be able to recreate your results from the submission for your group to qualify for the competition. Additionally, <strong>all tolerances (described in part 2) must still be met for your submission to qualify.</strong>

Your model should be a <em>.py </em>file and must adhere to the following naming procedure:

<em>{class_section}_{group_number}_model.py</em>, with no brackets. So if I was a CSE474 student in group #22, I would submit <em>474_22_model.py</em>.

When placed in the same directory as the data, your model file should act similarly to the models provided to you. It must call the <em>preprocessing </em>function with a metric list, load the data, build the model, make predictions, group those predictions by race, apply the postprocessing method of your choice, and print all relevant associated metrics and thresholds. You only need to print the results of your chosen postprocessing method, and thus you should not call <em>report_results </em>in your submitted model<em>. </em>You can use pieces of that code if you find it convenient. Any model that does not print its final results or whose results do not match those presented in the report will not be considered for the market competition.

Keep in mind that not all groups in the class have been assigned the same role. Some of you will be approaching this problem from very different perspectives, which will certainly affect which models and algorithms you choose. Open discussion and debate about what qualifies as fairness is encouraged on Piazza. However, conspiring as a class in an attempt to game the system (such as by all agreeing to use the same postprocessing algorithm) will not be tolerated and will result in the cancellation of any applicable extra credit points for this project.

Winning the market evaluation will result in 15pts extra credit for first place, 10pts extra credit for second place, and 5pts extra credit for third place. In the event of a tie for any position the winners will be determined by the strength of the argument provided in their report.




<h2>6 – Grading</h2>

Our methods for grading your submission are provided below:

<strong><em>Implementation </em></strong>– The 5 post-processing techniques described in section 2 are worth 10pts each.

<ul>

 <li>Your solutions must solve the optimization problems presented; hard-coding thresholds will result in no points.</li>

 <li>All tolerances must be observed where specified.</li>

 <li>Your methods will be applied as the post-processing steps of a standard model and data set, and there are definitive optimal numerical answers that should be provided by each function.</li>

 <li>Each method must perform secondary optimization using either accuracy or financial cost, and you must specify which metric you have chosen at the beginning of your report. <strong><em>Not choosing a metric or switching metrics between functions will result in a loss of points!</em></strong></li>

</ul>




<strong><em>Evaluation</em></strong> – Each of the 5 main questions in section 3 are worth 8pts.

<ul>

 <li>Report <strong>MUST </strong>include your model choice, postprocessing algorithm choice, secondary optimization criteria (cost or accuracy), the total cost of your system’s choices to society, and your overall accuracy. Reports without these criteria will not be graded.</li>

 <li>Answers should be well-thought out and reflect the results of a serious exploration of the problem and its various aspects.</li>

 <li>Single word or single sentence answers will not be accepted.</li>

 <li>All answers must be compounded into a coherent report justifying the choices made by your group.</li>

 <li>Satisfactory reports must not only be factually accurate, but also grammatically correct and persuasive.</li>

 <li>Additional 10pts come from professionalism and consistency – refer to section 3 for more details.</li>

</ul>




<strong>Extra Credit: </strong>

<ul>

 <li>The group with the best report will be awarded an extra 5pts.</li>

 <li>The winners of the market evaluation competition will be awarded an extra 15pts, 10pts, or 5pts for first, second, and third place respectively.</li>

</ul>




<h2>7 – Submission</h2>




Your group must submit a single zip file using the name {<em>class section}_{group number}_PA3.zip</em>, with no brackets. This zip file must contain the following files:




<ul>

 <li>Your report, in pdf format.</li>

 <li><em>py</em>, with all functions completed.</li>

 <li>A python file containing your market submission model.</li>

</ul>







<h2>8 – Required Libraries</h2>

You will need the following libraries to run the files for this project. The versions are not strictly necessary, but may provide some assistance when troubleshooting future compatibility issues.

<ul>

 <li>Numpy</li>

 <li>Sklearn</li>

 <li>Matplotlib</li>

 <li>Tensorflow 1.15.0</li>

 <li>Keras 2.2.4</li>

</ul>

<em>                                     </em>





