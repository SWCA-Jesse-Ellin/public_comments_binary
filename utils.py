import pandas as pd
import re
from tqdm import tqdm

def collapse_windows(df, source_text, startcol, endcol, labelcol, textcol, numbercol, step_size):
    collapsed_text = []
    collapsed_labels = []
    collapsed_starts = []
    collapsed_ends = []
    average_confidence = []
    letter_nums = []
    position = 0
    while position < len(df):
        local_confidence = [df["confidence"].iloc[position]]
        current_label = df[labelcol].iloc[position]
        collapsed_labels.append(df[labelcol].iloc[position])
        collapsed_starts.append(df[startcol].iloc[position])
        collapsed_ends.append(df[endcol].iloc[position])
        letter_nums.append(df["letter_num"].iloc[position])
        if df[labelcol].iloc[position+1] != current_label:
            average_confidence.append(local_confidence[0])
            position += 1
            continue
        position += 1
        while position < len(df) and df[labelcol].iloc[position] == current_label:
            collapsed_ends[-1] = df[endcol].iloc[position]
            local_confidence.append(df["confidence"].iloc[position])
            position += 1
        average_confidence.append(sum(local_confidence)/len(local_confidence))
    for i in range(len(collapsed_starts)):
        if collapsed_ends[i] < collapsed_starts[i]:
            collapsed_ends[i] = -1
        collapsed_text.append(source_text[letter_nums[i]][collapsed_starts[i]:collapsed_ends[i]])
    return pd.DataFrame({textcol : collapsed_text,
                         startcol : collapsed_starts,
                         endcol : collapsed_ends,
                         labelcol : collapsed_labels,
                         "letter_num" : letter_nums,
                         "confidence" : average_confidence})

def reconstruct_text(df, original_df, startcol, endcol, textcol, numbercol, window_size):
    original_text = {original_df[numbercol].iloc[i] : original_df[textcol].iloc[i] for i in range(len(original_df))}
    scrolling_index = {original_df[numbercol].iloc[i] : 0 for i in range(len(original_df))}
    new_text = []
    raw_text_indices = []
    for i in tqdm(range(len(df))):
        letter_num = df[numbercol].iloc[i]
        text = df[textcol].iloc[i]
        unprocessed_text = original_text[letter_num][scrolling_index[letter_num]::].lower()
        o_text = re.sub(r"\s+", " ", re.sub("[^a-zA-Z0-9]", " ", unprocessed_text))
        words = text.split(' ')
        start_index = [it.start() for it in re.finditer(words[0], string=o_text)]
        if len(words) != 1:
            for j in range(len(start_index)):
                split_text = o_text[start_index[j]::].split(' ')
                locations = [k for k in range(len(split_text)) if split_text[k] == words[1]]
                try:
                    if locations[0] < 5:
                        start_index = start_index[j]
                        break
                except:
                    continue
        if type(start_index) is list:
            if not start_index:
                start_index = [0]
            start_index = start_index[0]
        midpoint = start_index
        end_index = start_index
        for j, word in enumerate(words[1:]):
            options = [it.end() for it in re.finditer(pattern=word, string=unprocessed_text) if it.start() > end_index]
            if not options:
                continue
            end_index = options[0]
            if j == 5:
                midpoint = end_index + scrolling_index[letter_num]
        found_text = original_text[letter_num][start_index+scrolling_index[letter_num]:end_index+scrolling_index[letter_num]+1]
        raw_text_indices.append((start_index+scrolling_index[letter_num],end_index+scrolling_index[letter_num]+1))
        new_text.append(found_text)
        scrolling_index[letter_num] = midpoint
    df["original_text"] = new_text
    df["original_start"] = [ind[0] for ind in raw_text_indices]
    df["original_end"] = [ind[1] for ind in raw_text_indices]
    return df