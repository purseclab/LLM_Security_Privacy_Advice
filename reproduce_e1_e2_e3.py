import pandas as pd
import numpy as np
import warnings
from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sns.set_theme()
sns.set(font_scale=1.8)
warnings.filterwarnings("ignore")



class Experiment(Enum):
    E1 = "e1"
    E2 = "e2"
    E3 = "e3"

def process_experiment(exp):
    # Generates tables and plots for specific experiment
    print()
    print()
    print(f'Processing Experiment: {exp.value}')
    
    # Load dataset and responses
    questions = pd.read_csv("dataset/S&P-dataset.csv", encoding="windows-1252")
    df_chatGPT = pd.read_csv(f"experiments/ChatGPT-responses-{exp.value}.csv", encoding="windows-1252")
    df_bard = pd.read_csv(f"experiments/Bard-responses-{exp.value}.csv", encoding="windows-1252")



    # Preprocess
    df_chatGPT = pd.merge(questions[['id','category']], df_chatGPT, on='id', how='inner')
    df_bard = pd.merge(questions[['id','category']], df_bard, on='id', how='inner')

    df_chatGPT = df_chatGPT.rename(columns={"type" : "label"})
    df_bard = df_bard.rename(columns={"type" : "label"})

    desired_category_order = list(df_chatGPT.groupby('category').size().keys())
    

    label_types = ["Negate","Support","Unrelated","Noncommittal", "Partially Support"]

    def get_unique_types(row):
        # This function parses a row of a csv file and returns 
        # an integer representing the number of unique responses (labels)
        # from the LLM
        num_types = 0
        for type in label_types:
            if (row[type] > 0):
                num_types += 1
        return num_types

    def process(df):
        # This function parses a dataframe, grouping each response to generate 
        # a new dataframe 

        # Compute number of each response type
        new_df = pd.DataFrame(columns=['id','claim','category'] + label_types)
        # loop through every 5 rows in the original dataframe
        for i in range(0, len(df), 5):
            # slice the dataframe into a group of 5 rows
            newRow = dict(zip(label_types, np.zeros(len(label_types), dtype=int)))
            group = df.iloc[i : i+5]
            for j in range(5):
                newRow[group.iloc[j]['label']] +=1 
                
            newRow['id'] = group.iloc[0]['id']
            newRow['claim'] = group.iloc[0]['claim']
            newRow['category']=  group.iloc[0]['category']
            new_df = pd.concat([new_df, pd.DataFrame([newRow])], ignore_index=True)
  
  
        # Compute number of answered wrong or fully correctly questions 
        num_correct = 0
        num_wrong = 0
        five_responses_res = []
        # Iterate through each row of new df
        #new_df.to_csv('check.csv', index=False)
        for ind, row in new_df.iterrows():
            if (row['Negate'] == 5):
                five_responses_res.append("Correct")
                num_correct += 1
            # For false claim, if any support, consider incorrect
            # Or for true claim, if any negation, consider incorrect
            elif (row['Support'] > 0):
                five_responses_res.append("Incorrect")
                num_wrong += 1
            # Other response patterns considered as others
            else:
                five_responses_res.append("Others")
        new_df['result type for five'] = five_responses_res
       
        # Compute number of unique types
        new_df['unique label types'] = new_df.apply(get_unique_types, axis=1)

        return new_df


    if exp == Experiment.E2 or exp == Experiment.E3:
        df_chatGPT = process(df_chatGPT)

        # print(table)
        df_bard = process(df_bard)

    df_chatGPT['model'] = "ChatGPT"
    df_bard['model'] = "Bard"




    df_dict = {"ChatGPT": df_chatGPT, "Bard" : df_bard}
    df_both = pd.concat([df_chatGPT, df_bard])


    def printUniqueTable(someDf,dfColumnName,columnName):
        # prints a table statistics for
        # chatgpt and bard based on the specific column name
        # from the dataframe

        unique_answers_count = someDf[dfColumnName].value_counts().sort_index()
        # Calculate the percentage distribution
        total_rows = len(someDf)
        percentage_distribution = (unique_answers_count / total_rows) * 100

        # Create a DataFrame to display the results
        table = pd.DataFrame({columnName: unique_answers_count.index, 'Percentage': percentage_distribution.values})

        print(table.to_string(index=False))

    def showStat():
        # Calling functions to print statistics (tables in the paper)
        if exp == Experiment.E1:
            print("Table 4\n========")
            print()
            print("Bard")
            print()
            percentage_correct = (df_bard['label'].value_counts(normalize=True) * 100).reset_index()
            percentage_correct.columns = ['Unique Value', 'Percentage']
            print(percentage_correct.to_string(index=False))

            print()
            print("ChatGPT")
            print()
            percentage_correct = (df_chatGPT['label'].value_counts(normalize=True) * 100).reset_index()
            percentage_correct.columns = ['Unique Value', 'Percentage']
            print(percentage_correct.to_string(index=False))
            print()

        if exp == Experiment.E2:
            print("Table 5\n========")
            print()
            print("Bard")
            printUniqueTable(df_bard,'unique label types','Unique Responses')
            print()
            print("ChatGPT")
            printUniqueTable(df_chatGPT,'unique label types','Unique Responses')
            print()
            print("Table 6\n========")
            print()
            print("Bard")
            print()
            printUniqueTable(df_bard,'result type for five','Correctness')
            print()
            print("ChatGPT")
            print()
            printUniqueTable(df_chatGPT,'result type for five','Correctness')


        if exp == Experiment.E3:
            print("Table 7\n========")
            print()
            print("Bard")
            print()
            printUniqueTable(df_bard,'unique label types','Unique Responses')

            print()
            print("ChatGPT")
            print()
            printUniqueTable(df_chatGPT,'unique label types','Unique Responses')


            print("Table 8\n========")
            print()
            print("Bard")
            print()
            printUniqueTable(df_bard,'result type for five','Correctness')
            print()
            print("ChatGPT")
            print()
            printUniqueTable(df_chatGPT,'result type for five','Correctness')

           

    def plot():

        category_claim_count_dict = df_both.groupby(['model', 'category']).size()
        proportion_data = df_both.groupby(['model', 'category', which_proportion]).size().reset_index(name='claim_count')
        proportion_data['Proportion'] = proportion_data.apply(lambda row : row['claim_count']/category_claim_count_dict[row['model']][row['category']], axis=1)

        category_label_map = {category: i + 1 for i, category in enumerate(desired_category_order)}
        proportion_data['Category Label'] = proportion_data['category'].map(category_label_map)
        category_label_order = [category_label_map[category] for category in desired_category_order]

        ####
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['font.size'] = 25


        plot = sns.catplot(data=proportion_data, x="Category Label", y="Proportion", hue=which_proportion,
                    col="model", kind="bar", height=6, aspect=1.2, legend_out=False, order=category_label_order)
        plot.fig.set_size_inches(14,5)
        plot.fig.set_dpi(250)

        # Set legend
        colors = sns.color_palette()[:4]
        patterns = ['/', '+', '.', '-']
        legend_patterns = [Patch(facecolor=color, hatch=patch) for patch, color in zip(patterns, colors)]
        labels = plot.axes.flat[0].get_legend_handles_labels()
        plt.legend(handles=legend_patterns, labels=labels[1], bbox_to_anchor=(1, 1), loc='upper left')

        for ax in plot.axes.flat:
            for i, p in enumerate(ax.patches):
                p.set_hatch(patterns[int(i / len(desired_category_order))])

        # Set the title to model name
        plot.set_titles(col_template="{col_name}")
        # Set the actual column names or titles
        figure_titles = ['Bard', "ChatGPT"]
        for ax, title in zip(plot.axes.flat, figure_titles):
            ax.set_title(title)

        plt.xlabel('Category Label')
        plt.ylabel('Proportion')

        # Add category indicators
        legend_ax = plot.axes.flatten()[0]
        custom_legend = [plt.Line2D([], [], linestyle="", label=f'{category_label_map[category]}: {category}') for category in category_label_map]
        legend = legend_ax.legend(handles=custom_legend, loc='lower left', bbox_to_anchor=(-0.15, 1.11), ncol = int(len(category_label_order)/2), fontsize=18, frameon=False)

    
        plot.savefig(figure_name+'.pdf', dpi=250)

        for ax in plot.axes.flatten():
            for p in ax.patches:
                height = p.get_height()
                # Add the text annotation on top of each bar
                ax.annotate(f"{height * 100:.2f}%", (p.get_x() + p.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=10)

        plot.savefig(figure_name+'-with-number'+'.pdf', dpi=250)


    if exp == Experiment.E1:
        showStat()
        sns.set_palette("Set2")
        which_proportion = 'label'
        figure_name = f'plots/{exp.value}-res-label-per-category-normalized-count'
        plot()


    elif exp == Experiment.E2 or exp == Experiment.E3:
        # e2, e3 result type for five
        showStat()
        sns.set_palette("Set2")
        which_proportion = 'result type for five'

        figure_name = f'plots/{exp.value}-result-type-for-five-per-category-normalized-count'
        plot()
        # # e2, e3 unique label types
        sns.set_palette("Paired")
        which_proportion = 'unique label types'

        figure_name = f'plots/{exp.value}-unique-label-types-per-category-normalized-count'
        plot()

def main():
    process_experiment(Experiment.E1)
    process_experiment(Experiment.E2)
    process_experiment(Experiment.E3)

main()
