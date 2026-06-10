### Possible Reasons why the model might not learn on concepts with high scores (08-06-26)

- Relevance contamination in Attention pool i.e. higher order interactions,
- Self cancelling effects, usually prevalent in medical and financial timeseries data,
- cross cancellation among other concepts,

### Why the concepts chosen? Justification for the choice of concepts (08-06-26)

- Case by case analysis of the different properties of timeseries data. How prevalent are these concepts in real world data? How do they affect the learning of the model? Are there any specific patterns or characteristics in the data that make these concepts particularly relevant? In medical data, financial data, manufacturing data etc. Argue why the projection unto the vector space spanned by the concept vector make sense.
