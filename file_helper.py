import os
import pickle
import re
from tika import parser


def convert_pdf_to_pkl(pdf_folder, pkl_file_path):
    """ Load all pdfs in pdf_folder and save them as a single pkl file. """
    Years = [str(year) for year in range(1995, 2021)]
    Months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    Journal_list = ['RFS', 'JOF', 'JFE']
    docs = {}

    for j_name in Journal_list:
        docs[j_name] = {}

        for year in Years:
            docs[j_name][year] = {}

            for month in Months:
                docs[j_name][year][month] = []
                doc_folder = os.path.join(pdf_folder, j_name, year, month)

                if not os.path.isdir(doc_folder):
                    continue

                for file in os.listdir(doc_folder):
                    if not file.endswith('.pdf'):
                        continue
            
                    print(j_name, year, month, file)
                    doc = parser.from_file(os.path.join(doc_folder, file))['content']
                    doc = re.sub(r'"','fi',doc)
                    doc = re.sub(r'$','ffi',doc)
                    doc = re.sub(r'!','ff',doc)
                    docs[j_name][year][month].append(doc)

    with open(pkl_file_path, 'wb') as f:
        pickle.dump(docs, f)


def script_save_pkls():
    """ Convert pdf files to binary .pkl file.
        The downloaded pdf files were seperated by 5 parts.
    """

    pdf_folder = '/mnt/projects/COMP755/Data-20211009T234343Z-001/Data'
    pkl_file_path = '/mnt/projects/COMP755/001.pkl'
    convert_pdf_to_pkl(pdf_folder, pkl_file_path)

    pdf_folder = '/mnt/projects/COMP755/Data-20211009T234343Z-002/Data'
    pkl_file_path = '/mnt/projects/COMP755/002.pkl'
    convert_pdf_to_pkl(pdf_folder, pkl_file_path)

    pdf_folder = '/mnt/projects/COMP755/Data-20211009T234343Z-003/Data'
    pkl_file_path = '/mnt/projects/COMP755/003.pkl'
    convert_pdf_to_pkl(pdf_folder, pkl_file_path)

    pdf_folder = '/mnt/projects/COMP755/Data-20211009T234343Z-004/Data'
    pkl_file_path = '/mnt/projects/COMP755/004.pkl'
    convert_pdf_to_pkl(pdf_folder, pkl_file_path)

    pdf_folder = '/mnt/projects/COMP755/Data-20211009T234343Z-005/Data'
    pkl_file_path = '/mnt/projects/COMP755/005.pkl'
    convert_pdf_to_pkl(pdf_folder, pkl_file_path)


def script_merge_pkls():
    """ Merge the five .pkl files into a single .pkl file """

    folder = '/mnt/projects/COMP755'

    with open(os.path.join(folder, '001.pkl'), 'rb') as f:
        docs_1 = pickle.load(f, encoding='bytes')

    with open(os.path.join(folder, '002.pkl'), 'rb') as f:
        docs_2 = pickle.load(f, encoding='bytes')

    with open(os.path.join(folder, '003.pkl'), 'rb') as f:
        docs_3 = pickle.load(f, encoding='bytes')

    with open(os.path.join(folder, '004.pkl'), 'rb') as f:
        docs_4 = pickle.load(f, encoding='bytes')

    with open(os.path.join(folder, '005.pkl'), 'rb') as f:
        docs_5 = pickle.load(f, encoding='bytes')

    merged_docs = {}
    for j_name in docs_1.keys():
        merged_docs[j_name] = {}
        for year in docs_1[j_name].keys():
            merged_docs[j_name][year] = {}
            for month in docs_1[j_name][year].keys():
                merged_docs[j_name][year][month] = \
                    docs_1[j_name][year][month] + \
                    docs_2[j_name][year][month] + \
                    docs_3[j_name][year][month] + \
                    docs_4[j_name][year][month] + \
                    docs_5[j_name][year][month]

    with open(os.path.join(folder, 'data.pkl'), 'wb') as f:
        pickle.dump(merged_docs, f)


def load_data(pkl_file_path):
    """ Load .pkl file as a dictionary. """

    with open(pkl_file_path, 'rb') as f:
        docs = pickle.load(f, encoding='bytes')

    return docs
