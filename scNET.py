#!/usr/bin/env python
# coding: utf-8

import gdown
import os
import scanpy as sc
import scNET
import seaborn as sns
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("scNET is running")

    download_url = f'https://drive.google.com/uc?id=1C_G14cWk95FaDXuXoRw-caY29DlR9CPi'
    output_path = './example.h5ad'
    if not os.path.exists(output_path):
        gdown.download(download_url, output_path, quiet=False)

    # For faster processing in medium to large datasets (e.g. 30K or above cells), the maximum cells batch size can be increased depending on the available GPU memory.

    # For GPU with 24GB memory
    scNET.main.MAX_CELLS_BATCH_SIZE = 1000

    # for GPU with 40GB memory
    # scNET.main.MAX_CELLS_BATCH_SIZE = 4000

    # For GPU with 80GB memory or more
    # scNET.main.MAX_CELLS_BATCH_SIZE = 8000

    # otherwize, do not change the default value

    # To control the cutoff of gene expression, the minimum precetage of cells expressing a gene can be adjusted. The default all expressed genes are considered.
    # For example, to consider genes expressed in at least 5% of cells
    # scNET.main.EXPRESSION_CUTOFF = 0.05

    # For larger dataset (10K or above), containing larger number of subcommunities, the number of encoder layers could be increased to 4 or more. the default value is 3.
    scNET.main.NUM_LAYERS = 3

    obj = sc.read_h5ad("./example.h5ad")
    # scNET.run_scNET(obj, pre_processing_flag=False, human_flag=False, number_of_batches=10,
    #                 split_cells=True, max_epoch=300, model_name="test")

    # # Using the model's outputs

    # **Load all the relevant embeddings**
    #

    # In[ ]:

    embedded_genes, embedded_cells, node_features, out_features = scNET.load_embeddings("test")

    # **Create Scanpy object based on the reconstructed gene expression**
    #

    # In[ ]:

    cell_types = {"0": "Macrophages", "1": "Macrophages", "2": "CD8 Tcells", "3": "Microglia", "4": "Cancer",
                  "5": "CD4 Tcells", "6": "B Cells", "10": "Prolifrating Tcells", "8": "Cancer", "11": "NK"}
    obj.obs["Cell Type"] = obj.obs.seurat_clusters.map(cell_types)
    recon_obj = scNET.create_reconstructed_obj(node_features, out_features, obj)

    # **Plot marker genes**

    # In[ ]:

    sc.pl.umap(recon_obj, color=["Cell Type", "Cd4", "Cd8a", "Cd14", "Icos", "P2ry12", "Mki67", "Ncr1"], show=True,
               legend_loc='on data')

    # **Propagation based signature projection for actvation of Tcells**
    #
    #

    # In[ ]:

    scNET.run_signature(recon_obj, up_sig=["Zap70", "Lck", "Fyn", "Cd3g", "Cd28", "Lat"], alpha=0.9)

    # **And for Tumor aggression**

    # In[ ]:

    scNET.run_signature(recon_obj, up_sig=["Cdkn2a", "Myc", "Pten", "Kras"])

    # **Creating the co-embedded network, is it modular?**

    # In[ ]:

    net, mod = scNET.build_co_embeded_network(embedded_genes, node_features)
    print(f"The network mdularity: {mod}")

    # ##  Finding Downstream Transcription factors

    # ### Re-embed the T-cells subset  # TODO: why and how?

    # In[ ]:

    sub_obj = obj[obj.obs["Cell Type"] == "CD8 Tcells"]
    print('Re-embedding the T-cells subset')
    # scNET.run_scNET(sub_obj, pre_processing_flag=False, human_flag=False, number_of_batches=3, split_cells=False,
                    # max_epoch=300, model_name="Tcells")
    embedded_genes, embedded_cells, node_features, out_features = scNET.load_embeddings("Tcells")
    net, mod = scNET.build_co_embeded_network(embedded_genes, node_features, 99.5)
    print(f"The network mdularity: {mod}")

    # ### Find downstream TF's for a spesific gene signature

    # In[ ]:

    tf_scores = scNET.find_downstream_tfs(net, ["Zap70", "Lck", "Fyn", "Cd3g", "Cd28", "Lat"]).sort_values(
        ascending=False).head(10)

    ax = sns.barplot(x=tf_scores.index, y=tf_scores.values, color='skyblue')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    ax.set_xlabel('TF')
    ax.set_ylabel('Scores')
    plt.show()

    # **Finding differential enriched pathways**
    # ### Can we see a difference in phenotype between Cancer, Microglia and Macrophages?
    #
    #
    #
    #
    #
    #
    #

    # In[ ]:

    recon_obj.obs["Cell Type"] = recon_obj.obs.seurat_clusters.map(cell_types)
    de_genes_per_group, significant_pathways, filtered_kegg, enrichment_results = scNET.pathway_enricment(
        recon_obj.copy()[recon_obj.obs["Cell Type"].isin(["Microglia", "Macrophages", "Cancer"])], groupby="Cell Type")
    scNET.plot_de_pathways(significant_pathways, enrichment_results, 10)
