## Discussion Conclusion

- The current evaluation framework lacks comprehensive evaluation sets, limiting a standard to assess model performance in actual situations.
- ESM-3 foundation models present an opportunity to leverage both sequence and structure information for enhanced predictions.
- Integration of GNN with sequence-based models.

## TODO LIST

1. **ESM-3 Foundation Model Integration**
    - Use ESM-3 (both sequence and structure features)
    - Implement two pipeline architectures:
        - Cross-attention + MLP approach
        - MAE (dimensionality reduction) + MLP approach
    - Test on current `b4ppi` dataset (need to figure out the data configuration - Analyze protein species distribution)

2. **Test on new evaluation set**

3. **Further Exploration**
    - Model explainability
    - Investigate missing/vague protein label circumstances (semi-supervised learning)


