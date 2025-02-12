# Run on the server with enough RAM
# nohup python preprocess-and-count.py > foo.out 2> foo.err &

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# RESULTS_FOLDER = '../results/'
RESULTS_FOLDER = ''

INPUT_FOLDER = '/gpfs01/berens/data/data/pubmed_processed/'

def load_data(start_year=2010):
    print('Loading data...', flush=True)

    df = pd.read_csv(INPUT_FOLDER + "pubmed_baseline_2025.zip")
    print(f'Found {len(df)} papers.', flush=True)  # 24814136
    
    df = df[(df.Year >= start_year) & (df.Year <= 2024)]
    print(f'Kept {len(df)} papers from {start_year}--2024.', flush=True)  # 15103887
    
    return df
    
    # OBSOLETE: Needed for merging PubMed daily updates
    # df1 = pd.read_csv(INPUT_FOLDER + "pubmed_landscape_data_2024_v2.zip")
    # df1_abstracts = pd.read_csv(INPUT_FOLDER + "pubmed_landscape_abstracts_2024.zip")
    # df2 = pd.read_csv(INPUT_FOLDER + "pubmed_daily_updates_2024_v2.zip")

    # print('Assembling the dataframe...', flush=True)

    # df1["AbstractText"] = df1_abstracts["AbstractText"]

    # df1.drop(columns=np.setdiff1d(df1.columns, df2.columns), inplace=True)

    # df = pd.concat((df1, df2))
    # df = df.groupby(["PMID"]).last()


def cleanup_abstracts_inplace(df):
    # Titles filter

    ind = df.Title.str.contains("Correction:", regex=False)
    ind |= df.Title.str.contains("Correction to:", regex=False)
    ind |= df.Title.str.contains("Erratum:", regex=False)
    ind |= df.Title.str.contains("Erratum to:", regex=False)
    ind |= df.Title.str.contains("Corrigendum:", regex=False)
    ind |= df.Title.str.contains("Corrigendum to:", regex=False)
    ind |= df.Title.str.contains("Retracted:", regex=False)
    ind |= df.Title == "Retraction"

    df.loc[ind, "AbstractText"] = ""

    print(
        f"Removing {np.sum(ind)} abstracts (corrections/errata/retractions/etc).\n", 
        flush=True
    )  # 3514

    # Abstracts filter

    to_replace = {
        "&ldquo;": {"&ldquo;": '"', "&rdquo;": '"'},
        "&lsquo;": {"&lsquo;": "'", "&rsquo;": "'"},
        "&nbsp;": {"&nbsp;": " "},
        "&shy;": {"&shy;": ""},
        "&mdash;": {"&mdash;": "---"},
        "&ndash;": {"&ndash;": "--"},
        "u2002": {"\\u2002": " "},
        "<p>": {"<p>": "", "</p>": ""},
        "<em>": {"<em>": "", "</em>": ""},
        "This article": {"^This article has been.*": ""},
        "This manuscript": {"^This manuscript has been.*": ""},
        "The above article": {
            "^The above article.*": "",
            ".*The above article, published online.*": "",
        },
        "http": {"^http.*": ""},
        "For complete details": {
            "\s*For complete details on the use and execution of this protocol.*": "",
        },
        "For further information": {
            "For further information please consult linked data\.*": "",
        },
        "Communicated by": {"\s*\(?Communicated by.{0,100}$": ""},
        "Graphical abstract": {"\.\s*Graphical abstract.*": "."},
        "GRAPHICAL ABSTRACT": {"\.\s*GRAPHICAL ABSTRACT.*": "."},
        "VIDEO ABSTRACT": {"\.\s*VIDEO ABSTRACT.*": "."},
        "Video Abstract": {"\s*Video Abstract Available\.*": ""},
        "MINI ABSTRACT": {"\.\s*MINI ABSTRACT.*": "."},
        "ABSTRACT": {
            "^ABSTRACT[:.]?\s*": "",
            "^Abstract ABSTRACT[:.]?\s*": "",
            "^.{0,200} ABSTRACT: ": "",
        },
        "Abstract": {"^Abstract:?\s*": ""},
        "CONSPECTUS": {"CONSPECTUS: ": ""},
        "THIS ARTICLE": {
            "\s*THIS ARTICLE HAD BEEN MADE AVAILABLE FREE OF CHARGE.*": ""
        },
        "Copyright ©": {"\s*Copyright ©.*": ""},
        " © ": {
            "\.\s*[^.]*[0-9]\. © [12].*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\. Pediatr Pulmonol.$": ".",
            "\. Lasers Surg.$": ".",
            "\s+© .*": "",
        },
        ".© ": {
            "\.\s*[^.]*[0-9]\.© [12].*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\. Pediatr Pulmonol.$": ".",
            "\. Lasers Surg.$": ".",
            "\s+\.© .*": "",
        },
        " ©20": {
            "\.\s*[^.]*[0-9]\. ©20.{0,20}$": ".",
            "\s+©20.{0,20}$": "",
            "\.\s*[^.]*[0-9]\. ©20[0-2][0-9] AACRSee.*": ".",
            "\s+©20[0-2][0-9] AACRSee.*": "",
        },
        "Wiley Periodicals Inc": {
            "\.\s*[^.]*[0-9]\.\s*Published [12][890][0-9][0-9] Wiley Periodicals Inc.*": ".",
            "\. [a-zA-Z]+\.$": ".",
            "\. [ab-zA-Z]+\.$": ".",
            "\. [abc-zA-Z]+\.$": ".",
            "\. [abcd-zA-Z]+\.$": ".",
            "\s*Published [12][890][0-9][0-9] Wiley Periodicals Inc.*": ".",
        },
        "doi": {
            "\s*doi:\s*10\.[0-9a-zA-Z\.\/\-]*\s*$": "",
            "\s*doi:\s*10\.[0-9a-zA-Z\.\/\-]*\s*\(.*\)\.?\s*$": "",
            "\s*http://dx\.doi\.org/10\.[0-9a-zA-Z\.\/\-]*\s*$": "",
        },
        "DOI": {"\s*DOI: http://dx.doi.org/[0-9a-zA-Z\.\/\-]*\s*$": ""},
        "PMID": {".*PubMed PMID: [0-9]*\.\s*": ""},
        "Epub": {"\sEpub.{0,10}[12][890][0-9][0-9]\.?\s*$": ""},
        "Level of Evidence": {"\s*Level of Evidence:?\s*[0-9IV].*": ""},
        "LEVEL OF EVIDENCE": {"\s*LEVEL OF EVIDENCE:?\s*[0-9IV].*": ""},
        "Technical Efficacy": {"\s*[0-9] Technical Efficacy: Stage [0-9].*": ""},
        "Geriatr": {
            "\s*Geriatr Gerontol Int [\s0-9,;:\-\.\(\)]*$": "",
            "\s*J Am Geriatr Soc [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Genet Med": {"\s*Genet Med [\s0-9,;:\-\.\(\)]*$": ""},
        "Ann Neurol": {
            # Sometimes occurs twice
            "\s*Ann Neurol [\s0-9,;:\-\.\(\)]*$": "",
            "\s*Ann Neurol [\s01-9,;:\-\.\(\)]*$": "",
        },
        "ANN NEUROL": {"\s*ANN NEUROL [\s0-9,;:\-\.\(\)]*$": ""},
        "J Drugs Dermatol": {"\s*J Drugs Dermatol\. [\s0-9,;:\-\.\(\)]*$": ""},
        "Infect Control Hosp": {
            "\s*Infect Control Hosp Epidemiol [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Magn. Reson.": {
            "\.\s*[0-9]\s*J\. Magn\. Reson\. Imaging [\s0-9,;:\-\.\(\)]*$": ""
        },
        "MAGN. RESON.": {
            "\.\s*[0-9]\s*J\. MAGN\. RESON\. IMAGING [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Magnetic Resonance": {
            "\s*Magnetic Resonance in Medicine published by Wiley Periodicals\.*": "",
        },
        "(Pediatr Dent": {"\s*\(Pediatr Dent 20.*": ""},
        "Environ Toxicol Chem": {"\s*Environ Toxicol Chem [\s0-9,;:\-\.\(\)]*$": ""},
        "Environ Health Perspect": {
            "\s*Environ Health Perspect [\s0-9,;:\-\.\(\)]*$": ""
        },
        "Antioxid. Redox Signal.": {
            "\s*Antioxid\. Redox Signal\. [\s0-9,;:\-\.\(\)]*$": ""
        },
        "J Orthop Sports Phys Ther": {
            "\s*J Orthop Sports Phys Ther\.? [\sA0-9,;:\-\.\(\)]*$": ""
        },
        "J Strength Cond Res": {
            ".*J Strength Cond Res.{0,20}20[012][0-9]-([A-Z])": "\\1"
        },
        "Turk J Pediatr": {".*Turk J Pediatr [\s0-9,;:\-\.\(\)]*([A-Z])": "\\1"},
        "Laryngoscope": {
            "\s*[1-9][a-zA-Z]?\. Laryngoscope[^\.]*\.\s*$": "",
            "[\.][^\.]*Laryngoscope[^\.]*\.\s*$": ".",
            "\s*N/A\.$": "",
        },
        "Indian J Crit Care Med": {
            # Author list. Title. Indian J Crit Care Med
            "\s*[^\.]*\.[^\.]*[\.?!] Indian J Crit Care Med [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Int J Clin Pediatr Dent": {
            # Author list. Title. Int J Clin Pediatr Dent
            "\s*[^\.]*\.[^\.]*[\.?!] Int J Clin Pediatr Dent [\s0-9,;:\-\.\(\)]*$": "",
        },
        "J Clin Sleep Med": {
            # Author list. Title. Int J Clin Pediatr Dent
            "\s*[^\.]*\.[^\.]*[\.?!] J Clin Sleep Med\. [\s0-9,;:\-\.\(\)]*$": "",
        },
        "Hepatology": {
            "\(Hepatology [\s0-9,;:\-\.\(\)]*\)\.?\s*$": "",
            "\(Hepatology Communications [\s0-9,;:\-\.\(\)]*\)\.?\s*$": "",
        },
        "AASLD": {"[^\.]*AASLD\.\s*$": ""},
        "Database Record": {
            "\s*\(PsycINFO Database Record.*": "",
            "\s*\(PsycInfo Database Record.*": "",
        },
        "advance online publication": {
            "[^\.]* advance online publication,?\s*[0-9][0-9] [A-Za-z]* [0-9]{4}[\.;,]?\s*$": ""
        },
        "This article is protected": {"\sThis article is protected by copyright.*": ""},
        "This article is part": {
            "\s*This article is part of the themed issue.*": "",
            "\s*This article is part of a themed issue.*": "",
            "\s*This article is part of a themed section.*": "",
            "\s*This article is part of a Special Issue entitled.*": "",
        },
        "Elsevier Ltd": {"\s*20[012][0-9] Elsevier Ltd.*": ""},
        "How to cite this article:": {"\s*How to cite this article:.*": ""},
        "Cite this article:": {"\s*Cite this article:.*": ""},
        "Citation": {"\s+Citation: .*": ""},
        "ClinicalTrials.gov:": {"\s*\(?ClinicalTrials.gov: .{0,100}$": ""},
        ".].": {"\[[^\[]*[0-9]\.\]\.?$": ""},
        "https://youtu.be": {
            "\.\s*[^.]*: https://youtu.be.{0,100}$": ".",
            "\. https://youtu.be.{0,50}.$": ".",
        },
        "The virtual slide(s) for this article": {
            "\s*The virtual slide\(s\) for this article.*": ""
        },
        "IMPACT STATEMENT": {"\.\s*IMPACT STATEMENT[A-Z].*": "."},
        "Impact statement": {"\.\s*Impact Statement[A-Z].*": "."},
        ("RESULTS", "CONCLUSIONS"): {
            "\.\s*PURPOSE[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*BACKGROUND[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*INTRODUCTION[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*OBJECTIVE[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*MATERIALS AND METHODS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*MATERIALS & METHODS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*METHODS?[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*METHODOLOGY[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*DESIGN[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*STUDY DESIGN[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*KEY RESULTS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*RESULTS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS AND INFERENCES[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*CONCLUSIONS & INFERENCES[.: ][.:]?\s*([A-Z])": ". \\1",
        },
        ("Results", "Conclusions"): {
            "\.\s*Purpose[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Background[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Introduction[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Objective[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials and methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials and Methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Materials & Methods[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Methods?[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Methodology[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Design[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Study Design[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Key Results[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Results[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions and inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions and Inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions & inferences[.: ][.:]?\s*([A-Z])": ". \\1",
            "\.\s*Conclusions & Inferences[.: ][.:]?\s*([A-Z])": ". \\1",
        },
        "Expert commentary": {
            "\.\s*Expert commentary:\s*": ". ",
            "\.\s*Areas covered:\s*": ". ",
        },
        "Details of funding": {"\s*Details of funding are provided.*": ""},
        "This journal requires": {"\s*This journal requires.*": ""},
        "Proprietary or commercial disclosure": {
            "\s*Proprietary or commercial disclosure.*": ""
        },
        "See acknowledgments": {".\s*See acknowledgments.\s*$": "."},
        "This article is one of ten": {"\s*This article is one of ten reviews.*": ""},
        "In an effort to expedite the publication of articles": {
            "^In an effort to expedite the publication of articles.*": ""
        },
        "For complete coverage": {
            "\s*For complete coverage of all related areas of Endocrinology.*": ""
        },
        "Abbreviations": {
            "\.\s*Abbreviations:.*": ".",
            "\.\s*Abbreviations [Uu]sed:.*": ".",
        },
        "ABBREVIATIONS": {
            "\.\s*ABBREVIATIONS:.*": ".",
            "\.\s*ABBREVIATIONS USED:.*": ".",
        },
        "Registration number": {"\s*Registration number of the clinical trial:.*": ""},
        " de ": {
            "\.\s*:?\s*[Aa]nalisar .*": ".",
            "\.\s*:?\s*[Dd]escrever .*": ".",
            "\.\s*:?\s*[Ii]mplementar .*": ".",
            "\.\s*:?\s*[Cc]ompreender .*": ".",
            "\.\s*:?\s*[Aa]valiar .*": ".",
            "\.\s*:?\s*[Ee]stimar .*": ".",
            "\.\s*:?\s*[Dd]eterminar .*": ".",
            "\.\s*:?\s*[Rr]ealizar .*": ".",
            "\.\s*:?\s*[Cc]aracterizar .*": ".",
            "\.\s*:?\s*[Ii]dentificar .*": ".",
            "\.\s*:?\s*[Dd]iscutir .*": ".",
            "\.\s*:?\s*[Cc]onhecer .*": ".",
            "\.\s*:?\s*[Cc]onocer .*": ".",
            "\.\s*:?\s*Resumo .*": ".",
        },
    }

    all_affected_abstracts = np.zeros(len(df), dtype=bool)

    for search_string in to_replace:
        if type(search_string) == str:
            print(f"Searching for: {search_string}", end="", flush=True)
            ind = df.AbstractText.str.contains(search_string, regex=False)
        else:
            print("Searching for: " + " + ".join(search_string), end="", flush=True)
            ind = np.ones(len(df), dtype=bool)
            for search_str in search_string:
                ind &= df.AbstractText.str.contains(search_str, regex=False)

        print(f" --> found {np.sum(ind)} abstracts.", flush=True)

        for replace_string in to_replace[search_string]:
            s = (
                df[ind]
                .AbstractText.str.extract("(" + replace_string + ")")
                .values[:, 0]
            )

            ind2 = [type(ss) == str for ss in s.ravel()]
            all_affected_abstracts[np.where(ind)[0][ind2]] = True

            s = [ss[:75] for ss in s.ravel() if type(ss) == str]
            if len(s) > 0:
                print(
                    f"   Found {len(s)} abstract(s) with string(s) to replace:", 
                    flush=True
                )
                print("      " + "\n      ".join(s[:5]) + "\n", flush=True)
            else:
                print("   Found nothing to replace.\n", flush=True)
            df.loc[ind, "AbstractText"] = df[ind].AbstractText.str.replace(
                replace_string, to_replace[search_string][replace_string], regex=True
            )

    print(f"In total {np.sum(all_affected_abstracts)} were edited.", flush=True)  # 270189


def vectorize_abstracts(df):
    vectorizer = CountVectorizer(binary=True, min_df=1e-6)
    X = vectorizer.fit_transform(df.AbstractText.values)  # ~30 min

    print(f"Count matrix computed: {X.shape}", flush=True)  # 14448711 x 4179571

    pickle.dump(X, open(RESULTS_FOLDER + "counts.pkl", "wb"))

    words = vectorizer.get_feature_names_out()
    years = np.arange(2010, 2025)
    counts = np.zeros((words.size, years.size))
    totals = np.zeros(years.size)

    for i, year in enumerate(years):
        ind = df.Year == year
        counts[:, i] = np.array(np.sum(X[ind, :], axis=0)).ravel()
        totals[i] = np.sum(ind)

    df = pd.DataFrame(
        dict(zip(["word"] + list(years), [words] + list(counts.astype(int).T)))
    )
    df.loc[len(df)] = [""] + list(totals.astype(int))
    df.to_csv(RESULTS_FOLDER + "yearly-counts.csv.gz", index=False)

    return X, words, years, counts, totals
    

def compute_excess(targetYear, cutoff=1e-4):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    allowedWords =  np.array([np.all([s in alphabet for s in w]) for w in words])
    allowedWords &= np.array([len(w) >= 4 for w in words])
    
    freqs = (counts + 1) / (totals + 1)

    subsetWords = allowedWords & (
        freqs[:, years == targetYear].ravel() >= cutoff
    ) & (
        freqs[:, years == targetYear - 1].ravel() >= cutoff
    )

    projection = freqs[subsetWords, years == targetYear - 2] + np.maximum(
        (
            freqs[subsetWords, years == targetYear - 2] - 
            freqs[subsetWords, years == targetYear - 3]
        ) * 2,
        0
    )
    
    ratios = freqs[subsetWords, years == targetYear] / projection
    diffs  = freqs[subsetWords, years == targetYear] - projection
    current_freqs = freqs[subsetWords][:, years == targetYear].ravel()
    
    return subsetWords, ratios, diffs, current_freqs
    

def compute_excess_gaps():
    subsetWords, ratios, diffs, x = compute_excess(2024)

    ind = np.log10(ratios) > np.log10(2) - (np.log10(x) + 4) * (np.log10(2) / 4)
    ind |= diffs > 0.01

    annotations = pd.read_csv(RESULTS_FOLDER + 'excess_words.csv')
    word2type = dict(zip(annotations.word, annotations.type))

    chatgpt_words =   np.array(
        [w         for i, w in enumerate(words[subsetWords][ind]) if word2type[w] == 'style']
    )
    chatgpt_words_f = np.array(
        [x[ind][i] for i, w in enumerate(words[subsetWords][ind]) if word2type[w] == 'style']
    )

    cutoffs = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    cutoff_counts = np.zeros((len(cutoffs), years.size))
    
    for i, cutoff in enumerate(cutoffs):
        print('.', end='', flush=True)
        ind_words = np.isin(words, chatgpt_words[chatgpt_words_f < cutoff])
        for j, year in enumerate(years):
            ind = df.Year == year
            cutoff_counts[i, j] = np.sum(np.sum(X[ind, :][:, ind_words], axis=1) > 0)
    print('', flush=True)
    
    np.save(RESULTS_FOLDER + 'yearly-counts-cutoff.npy', cutoff_counts)
    

def compute_excess_gaps_subgroups(rare_threshold=0.01, output_filename="yearly-counts-subgroups.csv"):
    subsetWords, ratios, diffs, x = compute_excess(2024)

    ind = np.log10(ratios) > np.log10(2) - (np.log10(x) + 4) * (np.log10(2) / 4)
    ind |= diffs > 0.01

    annotations = pd.read_csv(RESULTS_FOLDER + 'excess_words.csv')
    word2type = dict(zip(annotations.word, annotations.type))

    chatgpt_words =   np.array(
        [w         for i, w in enumerate(words[subsetWords][ind]) if word2type[w] == 'style']
    )
    chatgpt_words_f = np.array(
        [x[ind][i] for i, w in enumerate(words[subsetWords][ind]) if word2type[w] == 'style']
    )

    chatgptwords_rare = chatgpt_words[chatgpt_words_f < rare_threshold]

    chatgptwords_common = [
        'exhibited', 'crucial', 'additionally', 'within', 'notably', 
        'insights', 'comprehensive', 'across', 'particularly', 'enhancing'
    ]

    ind_words_common = np.isin(words, chatgptwords_common)
    ind_words_rare = np.isin(words, chatgptwords_rare)

    def count(subset_papers):
        group_counts = np.zeros((3, years.size), dtype=int)
        for i, year in enumerate(years):
            ind = subset_papers & (df.Year == year)
            for j, ind_words in enumerate([ind_words_common, ind_words_rare]):
                group_counts[j, i] = np.sum(np.sum(X[ind, :][:, ind_words], axis=1) > 0)
            group_counts[2, i] = np.sum(ind)
        return group_counts

    def write_to_file(f, labeltype, label, group_counts):
        if "," in label:
            save_label = '"' + label + '"'
        else:
            save_label = label
        print(
            f'{labeltype},{save_label},' + ','.join(group_counts.ravel().astype(str)),
            file=f, 
            flush=True
        )
        print(
            f'{labeltype},{save_label},' + ','.join(group_counts.ravel().astype(str)),
            flush=True
        )

    with open(RESULTS_FOLDER + output_filename, 'w') as f:
        print(
            'grouptype,group,2010_common,2011_common,2012_common,2013_common,'
            '2014_common,2015_common,2016_common,2017_common,2018_common,'
            '2019_common,2020_common,2021_common,2022_common,2023_common,'
            '2024_common,2010_rare,2011_rare,2012_rare,2013_rare,2014_rare,'
            '2015_rare,2016_rare,2017_rare,2018_rare,2019_rare,2020_rare,'
            '2021_rare,2022_rare,2023_rare,2024_rare,2010_total,2011_total,'
            '2012_total,2013_total,2014_total,2015_total,2016_total,2017_total,'
            '2018_total,2019_total,2020_total,2021_total,2022_total,2023_total,'
            '2024_total',
            file=f,
            flush=True
        )

        ind = np.ones(len(df), dtype=bool)
        write_to_file(f, 'all', 'all', count(ind))

        for label in set(np.unique(df.Labels.values)) - set(['unlabeled']):
            ind = df.Labels == label
            write_to_file(f, 'class', label, count(ind))

        countries, countries_counts = np.unique(df.Countries.values, return_counts=True)
        countries = countries[np.argsort(countries_counts)][::-1]
        countries = [c for c in countries if c != 'unknown']
        for country in countries[:50]:
            ind = df.Countries == country
            write_to_file(f, 'country', country, count(ind))

        for gender in ['male', 'female']:
            ind = df.InferredGenderFirstAuthor == gender
            write_to_file(f, 'gender', gender + ' first', count(ind))

        for gender in ['male', 'female']:
            ind = df.InferredGenderLastAuthor == gender
            write_to_file(f, 'gender', gender + ' last', count(ind))

        journals, journals_counts = np.unique(df[df.Year == years[-1]].Journal.values, return_counts=True)
        journals = journals[np.argsort(journals_counts)][::-1]
        for journal in journals[:100]:
            ind = df.Journal == journal
            write_to_file(f, 'journal', journal, count(ind))

        ind = df.Journal.isin(['Nature', 'Science', 'Cell'])
        write_to_file(f, 'journals', 'Nature+Science+Cell', count(ind))

        # Established 2018 or earlier
        nature_journals = [
            'Nature aging', 'Nature astronomy', 'Nature biomedical engineering',
            'Nature biotechnology', 'Nature cardiovascular research', 'Nature catalysis',
            'Nature cell biology', 'Nature chemical biology', 'Nature chemistry',
            'Nature climate change', 'Nature communications', 'Nature computational science',
            'Nature digest', 'Nature ecology & evolution', 'Nature electronics',
            'Nature energy', 'Nature genetics', 'Nature geoscience', 'Nature human behaviour',
            'Nature immunology', 'Nature materials', 'Nature medicine', 'Nature methods',
            'Nature microbiology', 'Nature nanotechnology', 'Nature neuroscience',
            'Nature photonics', 'Nature physics', 'Nature plants', 
            'Nature structural & molecular biology', 'Nature sustainability'
        ]

        ind = df.Journal.isin(nature_journals)
        write_to_file(f, 'journals', 'Nature family', count(ind))

        ind = df.Journal.str.contains('Frontiers')
        write_to_file(f, 'journals', 'Frontiers', count(ind))

        ind = df.Journal.str.contains('Basel')
        write_to_file(f, 'journals', 'MDPI', count(ind))

        tadpole_journals = [
            'BioFactors (Oxford, England)',
            'Journal of cellular biochemistry',
            'Biomedicine & pharmacotherapy = Biomedecine & pharmacotherapie',
            'International immunopharmacology',
            'Experimental and molecular pathology',
            'Artificial cells, nanomedicine, and biotechnology',
            'Cellular physiology and biochemistry : international journal of experimental cellular physiology, biochemistry, and pharmacology',
            'International journal of immunopathology and pharmacology',
            'Brazilian journal of medical and biological research = Revista brasileira de pesquisas medicas e biologicas',
            'Die Pharmazie',
            'European review for medical and pharmacological sciences',
            'International journal of clinical and experimental pathology',
            'Neoplasma',
        ]

        ind = df.Journal.isin(tadpole_journals)
        write_to_file(f, 'journals', 'Tadpole journals', count(ind))

        ind = df.Title.str.contains('review') | df.Title.str.contains('Review')
        write_to_file(f, 'titles', 'Reviews', count(ind))

        for country in ['China', 'South Korea', 'Taiwan', 'Iran']:
            for label in ['computation', 'bioinformatics', 'material', 'healthcare', 'environment']:
                ind = (df.Countries == country) & (df.Labels == label)
                write_to_file(f, 'country/class', country + ' & ' + label, count(ind))

        for country in ['China', 'South Korea', 'Taiwan', 'Iran']:
            for journal in ['Cureus', 'Sensors (Basel, Switzerland)']:
                ind = (df.Countries == country) & (df.Journal == journal)
                write_to_file(f, 'country/journal', country + ' & ' + journal, count(ind))

        for country in ['China', 'South Korea', 'Taiwan', 'Iran']:
            for journal in ['Frontiers', 'Basel']:
                ind = (df.Countries == country) & df.Journal.str.contains(journal)
                write_to_file(f, 'country/journals', country + ' & ' + journal, count(ind))


##########################################################################

df = load_data()

cleanup_abstracts_inplace(df)

X, words, years, counts, totals = vectorize_abstracts(df)

# X = pickle.load(open("counts.pkl", "rb"))

# df = pd.read_csv(RESULTS_FOLDER + "yearly-counts.csv.gz")
# words = df.word.values[:-1].astype(str)
# years = df.columns[1:].astype(int)
# counts = df.values[:-1, 1:].astype(int)
# totals = df.values[-1, 1:].astype(int)

compute_excess_gaps()

compute_excess_gaps_subgroups(0.02, "yearly-counts-subgroups.csv")


## Below is a stand-along script to analyze the Covid frequency gap

# import pandas as pd
# import numpy as np
# import pickle
    
# df = pd.read_csv("yearly-counts.csv.gz")
# words = df.word.values[:-1].astype(str)
# years = df.columns[1:].astype(int)
# counts = df.values[:-1, 1:].astype(int)
# totals = df.values[-1, 1:].astype(int)

# X = pickle.load(open("counts.pkl", "rb"))

# df = pd.read_csv("/gpfs01/berens/data/data/pubmed_processed/pubmed_baseline_2025.zip")
# df = df[(df.Year >= 2010) & (df.Year <= 2024)]

covid_words = ["covid", "pandemic", "coronavirus", "sars"]
ind_covid_words = np.isin(words, covid_words)
group_counts = np.zeros((2, years.size), dtype=int)
for i, year in enumerate(years):
     ind = df.Year == year
     group_counts[0, i] = np.sum(np.sum(X[ind, :][:, ind_covid_words], axis=1) > 0)
     group_counts[1, i] = np.sum(ind)
print(group_counts)
f = (group_counts[0].astype(float) + 1) / (group_counts[1] + 1)
print(f * 100)
