# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Expirement Design for Data Science
# ## Group 26
# %% [markdown]
# ### Imports

# %%

import pandas as pd
import numpy as np
import os
import initialize
from rank_bm25 import BM25Okapi

#https://link.springer.com/chapter/10.1007%2F978-3-030-45442-5_7#Bib1

# %% [markdown]
# ### Load Preprocessed Data

# %%
trec = initialize.get_TREC45_dataset()

# %% [markdown]
# ### Step 1 - BM25 Search

# %%
trec


# %%
bm25 = BM25Okapi(trec['Description_tokenized'], k1=1.2, b=0.75)


# %%
bm25_res = [[x.Number, bm25.get_scores(x.Title_tokenized)] for x in trec.itertuples()]


# %%
bm25_idx_value = [[res[0], [[i, x] for i, x in enumerate(res[1]) if float(x) != 0]] for res in bm25_res]


# %%
arr = np.array(bm25_idx_value)

# %% [markdown]
# ### Normalize Values

# %%
val_only = [x[1] for y in arr[:,1] if len(y) > 1 for x in y]

min_val = min([x[1] for y in arr[:,1] if len(y) > 1 for x in y])
max_val = max([x[1] for y in arr[:,1] if len(y) > 1 for x in y])


# %%
bm25_search_formatted = pd.Series(arr[:,1], index=arr[:,0]).apply(pd.Series).stack()
bm25_search_formatted


# %%



# %%



