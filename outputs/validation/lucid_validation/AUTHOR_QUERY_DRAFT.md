# LUCID reproduction questions — draft, not sent

Subject: Reproducing the per-sample LUCID/TAALED results in Baese-Berk et al. (2021)

We are evaluating an automated language-sample analysis pipeline and would like to reproduce your released lexical-diversity results on the same LUCID recordings. We obtained the LUCID data through the SpeechBox data-use form and your OSF `TAALED_results.csv` and analysis code.

Could you share or clarify:

1. The cleaned `index1.txt`–`index240.txt` files used for TAALED, or the script that created them. We understand that filled/unfilled pauses and vocal noises were removed; the handling of fragments, repetitions, contractions and British pause spellings would help us match the inputs.
2. A mapping from those index files to the original recording filename, target tier/channel, and any analyzed time interval. Speaker/condition/scene metadata alone leaves two vocoder recordings per published key. For example, F3/beach has `LCD_F3F4B1clv3.TextGrid` and `LCD_F3F4B2clv2.TextGrid`.
3. The TAALED GUI version, tokenization/lemmatization/POS settings and language-model version used. We distinguish original MTLD from the moving-average variants.
4. Confirmation of the no-barrier source/tier mapping. For example, under the released naming conventions, M17/farm maps to `LCD_M17M18F4cv1.TextGrid` / Speaker A. Our current prepared-text pipeline counts 1,298 tokens, whereas published `index167.txt` has 461. This may reflect an input or mapping difference; we have not reassigned recordings by numerical similarity.

Our supplement reproduces the article's four Table 1 token averages within 0.01 word. We have retained all published rows and currently leave ambiguous mappings unscored. We want to distinguish implementation discrepancies from differences in the original inputs before making a reproduction claim.

Thank you for any files or guidance you can provide.

This is an unsent research-query draft. No message has been sent to the authors.
