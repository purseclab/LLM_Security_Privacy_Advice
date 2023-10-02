import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sns.set_theme()

sns.set(font_scale=1.8)
warnings.filterwarnings("ignore")

# Load dataset and responses
questions = pd.read_csv("dataset/S&P-dataset.csv", encoding="windows-1252")
df_chatGPT = pd.read_csv(f"experiments/ChatGPT-responses-e4.csv", encoding="windows-1252")
df_bard = pd.read_csv(f"experiments/Bard-responses-e4.csv", encoding="windows-1252")


df_chatGPT = pd.merge(questions[['id','category']], df_chatGPT, on='id', how='inner')
df_bard = pd.merge(questions[['id','category']], df_bard, on='id', how='inner')

df_chatGPT['model'] = "ChatGPT"
df_bard['model'] = "Bard"

validity_map = {False : "Invalid", True : "Valid"}
df_chatGPT['validity'] = df_chatGPT['valid?'].map(validity_map)
df_bard['validity'] = df_bard['valid?'].map(validity_map)

df_chatGPT = df_chatGPT.rename(columns={"content analysis" : "relevance"})
df_bard = df_bard.rename(columns={"content analysis" : "relevance"})

df_valid = df_chatGPT[df_chatGPT['valid?'] == True]
df_bard_valid = df_bard[df_bard['valid?'] == True]

desired_category_order = list(df_chatGPT.groupby('category').size().keys())

df_both = pd.concat([df_chatGPT, df_bard])


def plot():
    category_claim_count_dict = df_both.groupby(['model', 'category']).size()
    proportion_data = df_both.groupby(['model', 'category', which_proportion]).size().reset_index(name='url count')
    proportion_data['Proportion'] = proportion_data.apply(lambda row : row['url count']/category_claim_count_dict[row['model']][row['category']], axis=1)

    category_label_map = {category: i + 1 for i, category in enumerate(desired_category_order)}
    proportion_data['Category Label'] = proportion_data['category'].map(category_label_map)
    category_label_order = [category_label_map[category] for category in desired_category_order]

    ####
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Sans Serif'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 25

    plot = sns.catplot(data=proportion_data, x="Category Label", y="Proportion", hue=which_proportion,
                col="model", kind="bar", height=6, aspect=1.2, legend_out=False, order=category_label_order)
    plot.fig.set_size_inches(15,5)
    plot.fig.set_dpi(250)

    # Set title to be model name
    plot.set_titles(col_template="{col_name}")
    figure_titles = ['Bard', "ChatGPT"]
    for ax, title in zip(plot.axes.flat, figure_titles):
        ax.set_title(title)


    # Add patterns to bar and legend
    patterns = ['/', '+', '.', '-']
    for ax in plot.axes.flat:
        for i, p in enumerate(ax.patches):
            p.set_hatch(patterns[int(i / len(desired_category_order))])


    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel('Category Label')
    plt.ylabel('Proportion')

    legend_ax = plot.axes.flatten()[0]
    custom_legend = [plt.Line2D([], [], linestyle="", label=f'{category_label_map[category]}: {category}') for category in category_label_map]
    legend_ax.legend(handles=custom_legend, loc='lower left', bbox_to_anchor=(-0.2, 1.11), ncol = len(category_label_order)/2, fontsize=18, frameon=False)


    plot.savefig(figure_name+'.pdf', dpi=250)

    for ax in plot.axes.flatten():
        for p in ax.patches:
            height = p.get_height()
            # Add the text annotation on top of each bar
            ax.annotate(f"{height * 100:.2f}%", (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=10)

    plot.savefig(figure_name+'-with-number'+'.pdf', dpi=250)

which_proportion = 'validity'
sns.set_palette("Set2")
figure_name = 'plots/e4-valid-normalized-count'
df_both = pd.concat([df_chatGPT, df_bard])
plot()


which_proportion = 'relevance'
sns.set_palette("Paired")
figure_name = 'plots/e4-relevance-normalized-count'
df_both = pd.concat([df_valid, df_bard_valid])
plot()