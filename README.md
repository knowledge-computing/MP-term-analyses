# Racial Terminology Identification & Racial Covenant Classification

Mapping Prejudice project

## To Do
Keys:
- (TID) Work related to racial term identification
- (DCL) Works related to document classification
- (OCR) Work relevant with OCR correction -> lowest priority

### In Progress
- [ ] (DCL) Try TARS few-shot
    - [ ] (DCL) Measure maximum context window
    - [ ] (DCL) Measure maximum context window of data

### Up Next
- [ ] (OCR) Measure maximum context window of BART based OCR correction
- [ ] (OCR) Test accuracy of OCR correction (use GitHub data and original data)

### Completed
- [x] (TID) Create NER training pipeline using ModernBERT
- [x] (TID) Create pipeline that will convert CSV files to HF Dataset
    - [x] (TID) Tokenization
    - [x] (TID) NER tagging with Class labels
