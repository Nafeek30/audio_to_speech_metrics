##loading libraries and packages to plot and analyze data
library(languageR)
install.packages("tidyverse")
library(tidyr)
library(ggplot2)
library(dplyr)
library(lme4)

####SETTING UP LEXICAL DIVERSITY DATA####
##read in lexical diversity data##
lexdiv <- read.csv("/Users/mbaese/Downloads/LDLSMeasures/TAALED_results.csv", 
                      header = TRUE)
summary(lexdiv)

##make condition a factor instead of string
lexdiv$condition <- as.factor(lexdiv$condition)


##contrast coding
##Comparisons we want to make: Barrier vs. noBarrier; L2 vs. L1; L1 vs. each other
lexdiv$condition.contrasts <- lexdiv$condition
contrasts(lexdiv$condition.contrasts) <- cbind ("Barrier vs. No Barrier"=c(-.33, -.33, 1, -.33), "L2 vs. L1"=c(-.5, 1, 0, -.5), 
                                                "Noise vs. Vocoder"=c(-1, 0, 0, 1))

##create new columns based on the contrast coding above that we can use for analysis
lexdiv$bar_nb <- contrasts(lexdiv$condition.contrasts)[,1][as.numeric(lexdiv$condition.contrasts)]
lexdiv$L2_L1 <- contrasts(lexdiv$condition.contrasts)[,2][as.numeric(lexdiv$condition.contrasts)]
lexdiv$nois_voc<-contrasts(lexdiv$condition.contrasts)[,3][as.numeric(lexdiv$condition.contrasts)]


#########MEM REGRESSIONS FOR LEXICAL DIVERSITY FOR ALL WORDS#####
##Linear mixed models regression for type-token ratio, all words
ttr_aw.lmer <-lmer(simple_ttr_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv)
summary(ttr_aw.lmer)

ttr_aw1.lmer <-lmer(simple_ttr_aw ~ L2_L1 + nois_voc + (1|speaker), data=lexdiv)
anova(ttr_aw1.lmer,ttr_aw.lmer)

ttr_aw2.lmer <-lmer(simple_ttr_aw ~ bar_nb + nois_voc + (1|speaker), data=lexdiv)
anova(ttr_aw2.lmer,ttr_aw.lmer)

ttr_aw3.lmer <-lmer(simple_ttr_aw ~ bar_nb + L2_L1 + (1|speaker), data=lexdiv)
anova(ttr_aw3.lmer,ttr_aw.lmer)

##Linear regression for more complex TTR, all words
maas_ttr_aw.lmer <-lmer(maas_ttr_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv)
summary(maas_ttr_aw.lmer)

maas_ttr_aw1.lmer <-lmer(maas_ttr_aw ~ L2_L1 + nois_voc + (1|speaker), data=lexdiv)
anova(maas_ttr_aw1.lmer,maas_ttr_aw.lmer)

maas_ttr_aw2.lmer <-lmer(maas_ttr_aw ~ bar_nb + nois_voc + (1|speaker), data=lexdiv)
anova(maas_ttr_aw2.lmer,maas_ttr_aw.lmer)

maas_ttr_aw3.lmer <-lmer(maas_ttr_aw ~ bar_nb + L2_L1 + (1|speaker), data=lexdiv)
anova(maas_ttr_aw3.lmer,maas_ttr_aw.lmer)

##Linear regression for moving average TTR
mattr50_aw.lmer <-lmer(mattr50_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv)
summary(mattr50_aw.lmer)

mattr50_aw1.lmer <-lmer(mattr50_aw ~ L2_L1 + nois_voc + (1|speaker), data=lexdiv)
anova(mattr50_aw1.lmer,mattr50_aw.lmer)

mattr50_aw2.lmer <-lmer(mattr50_aw ~ bar_nb + nois_voc + (1|speaker), data=lexdiv)
anova(mattr50_aw2.lmer,mattr50_aw.lmer)

mattr50_aw3.lmer <-lmer(mattr50_aw ~ bar_nb + L2_L1 + (1|speaker), data=lexdiv)
anova(mattr50_aw3.lmer,mattr50_aw.lmer)

##Hypergeometric distribution
hdd42_aw.lmer <-lmer(hdd42_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv)
summary(hdd42_aw.lmer)
##Here the difference between L2 and the others is ns but other effects are significant

hdd42_aw1.lmer <-lmer(hdd42_aw ~ L2_L1 + nois_voc + (1|speaker), data=lexdiv)
anova(hdd42_aw1.lmer,hdd42_aw.lmer)

hdd42_aw2.lmer <-lmer(hdd42_aw ~ bar_nb + nois_voc + (1|speaker), data=lexdiv)
anova(hdd42_aw2.lmer,hdd42_aw.lmer)

hdd42_aw3.lmer <-lmer(hdd42_aw ~ bar_nb + L2_L1 + (1|speaker), data=lexdiv)
anova(hdd42_aw3.lmer,hdd42_aw.lmer)


##mtld original
mtld_aw.lmer <-lmer(mtld_original_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv)
summary(mtld_aw.lmer)

mtld_aw1.lmer <-lmer(mtld_original_aw ~ L2_L1 + nois_voc + (1|speaker), data=lexdiv)
anova(mtld_aw1.lmer,mtld_aw.lmer)

mtld_aw2.lmer <-lmer(mtld_original_aw ~ bar_nb + nois_voc + (1|speaker), data=lexdiv)
anova(mtld_aw2.lmer,mtld_aw.lmer)

mtld_aw3.lmer <-lmer(mtld_original_aw ~ bar_nb + L2_L1 + (1|speaker), data=lexdiv)
anova(mtld_aw3.lmer,mtld_aw.lmer)

##mtld moving average
mtld_ma_bi_aw.lmer <-lmer(mtld_ma_bi_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv)
summary(mtld_ma_bi_aw.lmer)

mtld_ma_bi_aw1.lmer <-lmer(mtld_ma_bi_aw ~ L2_L1 + nois_voc + (1|speaker), data=lexdiv)
anova(mtld_ma_bi_aw1.lmer,mtld_ma_bi_aw.lmer)

mtld_ma_bi_aw2.lmer <-lmer(mtld_ma_bi_aw ~ bar_nb + nois_voc + (1|speaker), data=lexdiv)
anova(mtld_ma_bi_aw2.lmer,mtld_ma_bi_aw.lmer)

mtld_ma_bi_aw3.lmer <-lmer(mtld_ma_bi_aw ~ bar_nb + L2_L1 + (1|speaker), data=lexdiv)
anova(mtld_ma_bi_aw3.lmer,mtld_ma_bi_aw.lmer)

##mtld moving average 2
mtld_ma_wrap_aw.lmer <-lmer(mtld_ma_wrap_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv)
summary(mtld_ma_wrap_aw.lmer)

mtld_ma_wrap_aw1.lmer <-lmer(mtld_ma_wrap_aw ~ L2_L1 + nois_voc + (1|speaker), data=lexdiv)
anova(mtld_ma_wrap_aw1.lmer,mtld_ma_wrap_aw.lmer)

mtld_ma_wrap_aw2.lmer <-lmer(mtld_ma_wrap_aw ~ bar_nb + nois_voc + (1|speaker), data=lexdiv)
anova(mtld_ma_wrap_aw2.lmer,mtld_ma_wrap_aw.lmer)

mtld_ma_wrap_aw3.lmer <-lmer(mtld_ma_wrap_aw ~ bar_nb + L2_L1 + (1|speaker), data=lexdiv)
anova(mtld_ma_wrap_aw3.lmer,mtld_ma_wrap_aw.lmer)


####SETTING UP LEXICAL SOPHISTICATION DATA####
lexsoph <- read.csv("/Users/mbaese/Downloads/LDLSMeasures/TAALES_results.csv", 
                    header = TRUE)
summary(lexsoph)

##make condition a factor instead of string
lexsoph$condition <- as.factor(lexsoph$condition)

##contrast coding
##Comparisons we want to make: Barrier vs. noBarrier; L2 vs. L1; L1 vs. each other
lexsoph$condition.contrasts <- lexsoph$condition
contrasts(lexsoph$condition.contrasts) <- cbind ("Barrier vs. No Barrier"=c(-.33, -.33, 1, -.33), "L2 vs. L1"=c(-.5, 1, 0, -.5), 
                                                "Noise vs. Vocoder"=c(-1, 0, 0, 1))

##create new columns based on the contrast coding above that we can use for analysis
lexsoph$bar_nb <- contrasts(lexsoph$condition.contrasts)[,1][as.numeric(lexsoph$condition.contrasts)]
lexsoph$L2_L1 <- contrasts(lexsoph$condition.contrasts)[,2][as.numeric(lexsoph$condition.contrasts)]
lexsoph$nois_voc<-contrasts(lexsoph$condition.contrasts)[,3][as.numeric(lexsoph$condition.contrasts)]

###############MIXED MODELS FOR LEXICAL SOPHISTICATION DATA#####
#Frequency
freq_aw.lmer <-lmer(BNC_Spoken_Freq_AW ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph)
summary(freq_aw.lmer)

freq_aw1.lmer <-lmer(BNC_Spoken_Freq_AW ~ L2_L1 + nois_voc + (1|speaker), data=lexsoph)
anova(freq_aw1.lmer,freq_aw.lmer)

freq_aw2.lmer <-lmer(BNC_Spoken_Freq_AW ~ bar_nb + nois_voc + (1|speaker), data=lexsoph)
anova(freq_aw2.lmer,freq_aw.lmer)

freq_aw3.lmer <-lmer(BNC_Spoken_Freq_AW ~ bar_nb + L2_L1 + (1|speaker), data=lexsoph)
anova(freq_aw3.lmer,freq_aw.lmer)

#Range
range_aw.lmer <-lmer(BNC_Spoken_Range_AW ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph)
summary(range_aw.lmer)

range_aw1.lmer <-lmer(BNC_Spoken_Range_AW ~ L2_L1 + nois_voc + (1|speaker), data=lexsoph)
anova(range_aw1.lmer,range_aw.lmer)

range_aw2.lmer <-lmer(BNC_Spoken_Range_AW ~ bar_nb + nois_voc + (1|speaker), data=lexsoph)
anova(range_aw2.lmer,range_aw.lmer)

range_aw3.lmer <-lmer(BNC_Spoken_Range_AW ~ bar_nb + L2_L1 + (1|speaker), data=lexsoph)
anova(range_aw3.lmer,range_aw.lmer)
###Here, weirdly, vocoder is the lowest and L2 is highest - other two are between


#BigramNormed
bigram_aw.lmer <-lmer(BNC_Spoken_Bigram_Normed_Freq ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph)
summary(bigram_aw.lmer)

bigram_aw1.lmer <-lmer(BNC_Spoken_Bigram_Normed_Freq ~ L2_L1 + nois_voc + (1|speaker), data=lexsoph)
anova(bigram_aw1.lmer,bigram_aw.lmer)

bigram2_aw.lmer <-lmer(BNC_Spoken_Bigram_Normed_Freq ~ bar_nb + nois_voc + (1|speaker), data=lexsoph)
anova(bigram2_aw.lmer,bigram_aw.lmer)

bigram3_aw.lmer <-lmer(BNC_Spoken_Bigram_Normed_Freq ~ bar_nb + L2_L1 + (1|speaker), data=lexsoph)
anova(bigram3_aw.lmer,bigram_aw.lmer)

#TrigramNormed
trigram_aw.lmer <-lmer(BNC_Spoken_Trigram_Normed_Freq ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph)
summary(trigram_aw.lmer)

trigram_aw1.lmer <-lmer(BNC_Spoken_Trigram_Normed_Freq ~ L2_L1 + nois_voc + (1|speaker), data=lexsoph)
anova(trigram_aw1.lmer,trigram_aw.lmer)

trigram2_aw.lmer <-lmer(BNC_Spoken_Trigram_Normed_Freq ~ bar_nb + nois_voc + (1|speaker), data=lexsoph)
anova(trigram2_aw.lmer,trigram_aw.lmer)

trigram3_aw.lmer <-lmer(BNC_Spoken_Trigram_Normed_Freq ~ bar_nb + L2_L1 + (1|speaker), data=lexsoph)
anova(trigram3_aw.lmer,trigram_aw.lmer)
##All NS

#BigramProportion
bigramprop_aw.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph)
summary(bigramprop_aw.lmer)

bigramprop1_aw.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ L2_L1 + nois_voc + (1|speaker), data=lexsoph)
anova(bigramprop1_aw.lmer,bigramprop_aw.lmer)

bigramprop2_aw.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ bar_nb + nois_voc + (1|speaker), data=lexsoph)
anova(bigramprop2_aw.lmer,bigramprop_aw.lmer)

bigramprop3_aw.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ bar_nb + L2_L1 + (1|speaker), data=lexsoph)
anova(bigramprop3_aw.lmer,bigramprop_aw.lmer)

#TrigramProportion
trigramprop_aw.lmer <-lmer(BNC_Spoken_Trigram_Proportion ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph)
summary(trigramprop_aw.lmer)

trigramprop1_aw.lmer <-lmer(BNC_Spoken_Trigram_Proportion ~ L2_L1 + nois_voc + (1|speaker), data=lexsoph)
anova(trigramprop1_aw.lmer,trigramprop_aw.lmer)

trigramprop2_aw.lmer <-lmer(BNC_Spoken_Trigram_Proportion ~ bar_nb + nois_voc + (1|speaker), data=lexsoph)
anova(trigramprop2_aw.lmer,trigramprop_aw.lmer)

trigramprop3_aw.lmer <-lmer(BNC_Spoken_Trigram_Proportion ~ bar_nb + L2_L1 + (1|speaker), data=lexsoph)
anova(trigramprop3_aw.lmer,trigramprop_aw.lmer)

######LOOKING AT ORDER EFFECTS IN LEXICAL DIVERSITY DATA#######
lexdiv_order <- read.csv("/Users/mbaese/Downloads/LDLSMeasures/TAALED_results_order.csv", 
                   header = TRUE)
summary(lexdiv_order)

##make condition a factor
lexdiv_order$condition <- as.factor(lexdiv_order$condition)
##make order a factor
lexdiv_order$order <- as.factor(lexdiv_order$order)


##contrast coding
##Comparisons we want to make: Barrier vs. noBarrier; L2 vs. L1; L1 vs. each other
lexdiv_order$condition.contrasts <- lexdiv_order$condition
contrasts(lexdiv_order$condition.contrasts) <- cbind ("Barrier vs. No Barrier"=c(-.33, -.33, 1, -.33), "L2 vs. L1"=c(-.5, 1, 0, -.5), 
                                                "Noise vs. Vocoder"=c(-1, 0, 0, 1))

##create new columns based on the contrast coding above that we can use for analysis
lexdiv_order$bar_nb <- contrasts(lexdiv_order$condition.contrasts)[,1][as.numeric(lexdiv_order$condition.contrasts)]
lexdiv_order$L2_L1 <- contrasts(lexdiv_order$condition.contrasts)[,2][as.numeric(lexdiv_order$condition.contrasts)]
lexdiv_order$nois_voc<-contrasts(lexdiv_order$condition.contrasts)[,3][as.numeric(lexdiv_order$condition.contrasts)]

##MEMs including order (no significant contribution of order in any measure)
ttr_aw_order.lmer <-lmer(simple_ttr_aw ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexdiv_order)
summary(ttr_aw_order.lmer)
ttr_aw_noorder.lmer <-lmer(simple_ttr_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv_order)
anova(ttr_aw_order.lmer,ttr_aw_noorder.lmer)

maas_ttr_aw_order.lmer <-lmer(maas_ttr_aw ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexdiv_order)
summary(maas_ttr_aw_order.lmer)
maas_ttr_aw_noorder.lmer <-lmer(maas_ttr_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv_order)
anova(maas_ttr_aw_order.lmer,maas_ttr_aw_noorder.lmer)

mattr50_aw_order.lmer <-lmer(mattr50_aw ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexdiv_order)
summary(mattr50_aw_order.lmer)
mattr50_aw_noorder.lmer <-lmer(mattr50_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv_order)
anova(mattr50_aw_order.lmer, mattr50_aw_noorder.lmer)

hdd42_aw_order.lmer <-lmer(hdd42_aw ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexdiv_order)
summary(hdd42_aw_order.lmer)
hdd42_aw_noorder.lmer <-lmer(hdd42_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv_order)
anova(hdd42_aw_order.lmer,hdd42_aw_noorder.lmer)

mtld_aw_order.lmer <-lmer(mtld_original_aw ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexdiv_order)
summary(mtld_aw_order.lmer)
mtld_aw_noorder.lmer <-lmer(mtld_original_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv_order)
anova(mtld_aw_order.lmer, mtld_aw_noorder.lmer)

mtld_ma_bi_aw_order.lmer <-lmer(mtld_ma_bi_aw ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexdiv_order)
summary(mtld_ma_bi_aw_order.lmer)
mtld_ma_bi_aw_noorder.lmer <-lmer(mtld_ma_bi_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv_order)
anova(mtld_ma_bi_aw_order.lmer,mtld_ma_bi_aw_noorder.lmer)

mtld_ma_wrap_aw_order.lmer <-lmer(mtld_ma_wrap_aw ~ bar_nb + L2_L1 + nois_voc + order +  (1|speaker), data=lexdiv_order)
summary(mtld_ma_wrap_aw_order.lmer)
mtld_ma_wrap_aw_noorder.lmer <-lmer(mtld_ma_wrap_aw ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexdiv_order)
anova(mtld_ma_wrap_aw_order.lmer, mtld_ma_wrap_aw_noorder.lmer)

######LOOKING AT ORDER EFFECTS IN LEXICAL SOPHISTICATION DATA#######
lexsoph_order <- read.csv("/Users/mbaese/Downloads/LDLSMeasures/TAALES_results_order.csv", 
                    header = TRUE)
summary(lexsoph_order)

##make condition a factor instead of string
lexsoph_order$condition <- as.factor(lexsoph_order$condition)
##make order a factor
lexsoph_order$order <- as.factor(lexsoph_order$order)

##contrast coding
##Comparisons we want to make: Barrier vs. noBarrier; L2 vs. L1; L1 vs. each other
lexsoph_order$condition.contrasts <- lexsoph_order$condition
contrasts(lexsoph_order$condition.contrasts) <- cbind ("Barrier vs. No Barrier"=c(-.33, -.33, 1, -.33), "L2 vs. L1"=c(-.5, 1, 0, -.5), 
                                                 "Noise vs. Vocoder"=c(-1, 0, 0, 1))

##create new columns based on the contrast coding above that we can use for analysis
lexsoph_order$bar_nb <- contrasts(lexsoph_order$condition.contrasts)[,1][as.numeric(lexsoph_order$condition.contrasts)]
lexsoph_order$L2_L1 <- contrasts(lexsoph_order$condition.contrasts)[,2][as.numeric(lexsoph_order$condition.contrasts)]
lexsoph_order$nois_voc<-contrasts(lexsoph_order$condition.contrasts)[,3][as.numeric(lexsoph_order$condition.contrasts)]

##MEMs including order (no significant contribution of order in any measure)
freq_aw_order.lmer <-lmer(BNC_Spoken_Freq_AW ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexsoph_order)
summary(freq_aw_order.lmer)
freq_aw_noorder.lmer <-lmer(BNC_Spoken_Freq_AW ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph_order)
anova(freq_aw_order.lmer, freq_aw_noorder.lmer)

range_aw_order.lmer <-lmer(BNC_Spoken_Range_AW ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexsoph_order)
summary(range_aw_order.lmer)
range_aw_noorder.lmer <-lmer(BNC_Spoken_Range_AW ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph_order)
anova(range_aw_order.lmer, range_aw_noorder.lmer)

bigram_aw_order.lmer <-lmer(BNC_Spoken_Bigram_Normed_Freq ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexsoph_order)
summary(bigram_aw_order.lmer)
bigram_aw_noorder.lmer <-lmer(BNC_Spoken_Bigram_Normed_Freq ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph_order)
anova(bigram_aw_order.lmer, bigram_aw_noorder.lmer)

trigram_aw_order.lmer <-lmer(BNC_Spoken_Trigram_Normed_Freq ~ bar_nb + L2_L1 + nois_voc + order +  (1|speaker), data=lexsoph_order)
summary(trigram_aw_order.lmer)
trigram_aw_noorder.lmer <-lmer(BNC_Spoken_Trigram_Normed_Freq ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph_order)
anova(trigram_aw_order.lmer,trigram_aw_noorder.lmer)

trigramprop_aw_order.lmer <-lmer(BNC_Spoken_Trigram_Proportion ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexsoph_order)
summary(trigramprop_aw_order.lmer)
trigramprop_aw_noorder.lmer <-lmer(BNC_Spoken_Trigram_Proportion ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph_order)
anova(trigramprop_aw_order.lmer,trigramprop_aw_noorder.lmer)

##One significant effect for bigram proportion
bigramprop_aw_order.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ bar_nb + L2_L1 + nois_voc + order + (1|speaker), data=lexsoph_order)
summary(bigramprop_aw_order.lmer)
bigramprop_aw_noorder.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ bar_nb + L2_L1 + nois_voc + (1|speaker), data=lexsoph_order)
anova(bigramprop_aw_order.lmer,bigram_aw_noorder.lmer)

bigramprop1_aw_order.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ L2_L1 + nois_voc + order + (1|speaker), data=lexsoph_order)
anova(bigramprop1_aw_order.lmer,bigramprop_aw_order.lmer)

bigramprop2_aw_order.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ bar_nb + nois_voc + order + (1|speaker), data=lexsoph_order)
anova(bigramprop2_aw_order.lmer,bigramprop_aw_order.lmer)

bigramprop3_aw_order.lmer <-lmer(BNC_Spoken_Bigram_Proportion ~ bar_nb + L2_L1 + order + (1|speaker), data=lexsoph_order)
anova(bigramprop3_aw_order.lmer,bigramprop_aw_order.lmer)
