
## Open Problems

1. [ ] Fix the warning obtained when running 

   ```
   > evaluator(model_name, model, tokenizer, task_specs[0])
   A decoder-only architecture is being used, but right-padding was detected! 
   For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
   ```

1. [ ] Model responses do not seem to be working for Task 8, inspect.

   ```
   > evaluator(model_name, model, tokenizer, task_specs[0])
   ```

1. [ ] Mistral 7B Instruct gives deterministic answers. Resolve.

2. [ ] Add other models to the pipeline  
    - [ ] Gemma  
    - [ ] Claude  
    - [ ] DeepSeek 7B Chat (needs 8 H100s)  
    - [ ] GPT 4o (needs API credits)

3. [ ] Our benchmark claim depends on the fact that we have a "ground truth".  
      It is important to evaluate whether comparing variables across datasets gives the same answer.  
      For instance, is the age distribution in Census/NHANES/GSS exactly the same?  
      I.e., would one dataset obtain a score of 100 on another? Need to implement.

4. [ ] Adding new tasks

5. [ ] In the previous iterations, different datasets were used, which are worth  
      translating to the current pipeline (and making them into tasks).  
    - [ ] Sex by Occupation – US Department of Labor  
    - [ ] Sex by College Degree – US Department of Education  
    - [ ] Sex by Arrest Type – FBI
