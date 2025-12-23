#from img2table.ocr import PaddleOCR
#from img2table.document import PDF
import pandas as pd
import fitz
import os
from img2table.document import Image
from img2table.ocr import EasyOCR
import numpy as np
import streamlit as st
from datetime import datetime
import io
ocr = EasyOCR(lang=["en"])
templates=""
#PADDLE_PDX_DISABLE_DEV_MODEL_WL=1
#ocr = PaddleOCR(lang="en",device="cpu") # Speciy the language
st.title("Data Extractor")
st.subheader("Application to Extract Statement from PDF's to Downloadable DataFrames")

############################# Sidebar ########################
st.sidebar.title("Guide")
st.sidebar.markdown("> Quality screenshot images taken from phone, laptop, etc is preferred")
st.sidebar.markdown("> Images captured with phone camera yields incomplete data")
st.sidebar.markdown("""> Images affected by artifacts including partial occulsion, distorted perspective, 
                    and complex background yields incomplete tables""")
st.sidebar.markdown(""" > Handwriting recognition on images containing tables will be significantly harder
                       due to infinite variations of handwriting styles and limitations of optical character recognition""")


###################### loading images #######################
uploaded_file = st.file_uploader("Choose an image | Accepted formats: only PDF", type=("pdf"))
if uploaded_file is not None:


    fl= uploaded_file.name.replace('.pdf','').replace('.Pdf','').replace('.PDF','')
    pdf_path=fl + ".pdf"
    csv_path = fl + ".csv"
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    if os.path.exists(csv_path):
        print('Output file available')
        temp_df=pd.read_csv(fl + '.csv',nrows=1)
        ncols = temp_df.shape[1]
        df = pd.read_csv(fl + '.csv', usecols=range(1, ncols),encoding='utf-8')
        cols_to_fill_indices = [0]
        cols_to_fill_labels = df.columns[cols_to_fill_indices]
        df[cols_to_fill_labels] = df[cols_to_fill_labels].ffill()
    else:
    #pdf = PDF(src="1830.pdf")
        dpi = 300 # Desired resolution
        zoom = dpi / 72 # Zoom factor, standard: 72 dpi
        magnify = fitz.Matrix(zoom, zoom) # Magnifies in x and y directions
        pdf_name = os.path.splitext(pdf_path)[0]
        #pdf_tables = pdf.extract_tables(ocr=ocr, implicit_rows=True, borderless_tables=True)
        df_fields = pd.DataFrame()
        page_set = fitz.open(stream=uploaded_file.read(),filetype="pdf")
        pg=0
        full_df=pd.DataFrame()
        for page in page_set:
                image_filename = f"{pdf_name}_page_{pg+1}.png"
                image_path = os.path.join('Images', image_filename)
                uploaded_file=image_path
                pix = page.get_pixmap(matrix=magnify) # Render page to an image
                output_buffer = io.BytesIO()
                pix.save(output_buffer, format="PNG") 
                print(f"  Saved: {image_filename}")
                img = Image(pix)
                extracted_tables = img.extract_tables(ocr=ocr, implicit_rows=True, implicit_columns = True, borderless_tables=True)
        # 4. Process the extracted tables (e.g., convert to pandas DataFrame and print)
                for table in extracted_tables:
                    #print(f"Table found with shape: {table.shape}")
                    # Get the table content as a pandas DataFrame
                    df = table.df
                    print(df)
                    ext_df = pd.DataFrame(df)
                    colname=ext_df.columns[0]
                    ext_df = ext_df[~ext_df[colname].str.contains('Date', na=False)]
                    full_df=pd.concat([full_df,ext_df])
                    # try:
                    #     cols_to_fill_indices = [0]
                    #     cols_to_fill_labels = ext_df.columns[cols_to_fill_indices]
                    #     ext_df[cols_to_fill_labels] = ext_df[cols_to_fill_labels].ffill()
                        
                    #     merged_df=ext_df
                        
                    #     #merged_df = ext_df.groupby([cols_to_fill_labels(0)])   

                    #     #print('Merged Data')
                    #     print(merged_df)      
                    # except:
                    #     try:
                    #         cols_to_fill_indices = [0, 2,3]
                    #         cols_to_fill_labels = ext_df.columns[cols_to_fill_indices]
                    #         ext_df[cols_to_fill_labels] = ext_df[cols_to_fill_labels].ffill()
                    #         merged_df=ext_df
                    #         #merged_df = ext_df.groupby([cols_to_fill_labels(0)])   
                    #         #print('Merged Data')
                    #         print(merged_df)  
                    #     except:
                    #         pass      
                    
                        
                        
                    
                
                pg=pg+1
        colname=full_df.columns[0]
        full_df.to_csv(fl + '_Without.csv',encoding='utf-8-sig')
        full_df[colname] = full_df[colname].fillna(method='ffill')
        print(full_df)
        full_df.to_csv(fl+'.csv',encoding='utf-8-sig')
        df=full_df

    #for page_num, tables_on_page in pdf_tables.items():
    #    print(f"--- Page {page_num + 1} ---")
    #    for table in tables_on_page:
    #        # Access the table as a pandas DataFrame
    #        df = table.df
    #        print(df)
    #        df_fields.append(df)
            
    #        df.to_csv(f"table_page_{page_num + 1}.csv", index=False)

    #df_fields.to_csv('1830.csv', index=False)

    df = df.replace('', np.nan)

    st.subheader("DataFrame")
    st.dataframe(df)
    debitname = df.columns[2]
    balance = df.columns[4]
    creditname = df.columns[3]
    decname = df.columns[1]
    dates = df.columns[0]
    df[decname] = df[decname].astype(str)
    df[dates] = df[dates].astype(str)
    years=2025
    #merged_df = df.groupby([cols_to_fill_labels[0]], as_index=False).agg(lambda x: ', '.join(x.dropna().astype(str)) if x.notna().any() else np.nan)
    #cols_to_fill_indices = [0,1,2]

    #df_cleaned = df[df[[df.columns[2],df.columns[3]]].ne('').all(axis=1)]
    column_names = ['Date', 'Description', 'Debit or Credit','Amount','Balance']
    final_df=pd.DataFrame(columns=column_names)
    print(final_df)
    a=0
    #df_filtered = df.loc[~df[decname].str.contains('forward', case=False)]
    #df=df_filtered
    templates=""
    for col_name, col_values in df.items():
        #print(f"Column name: {col_name}")
        for value in col_values:
            if "hsbcn" in str(value).lower():
                templates="HSBC"
            if "To Contact U.S. Bank" in str(value) or 'uslbank' in str(value) or '800-US BANKS' in str(value):
                templates="U.S. BANK"
            #print(str(value))

    print('Found Template of : ' + templates)
    #1830
    if templates=="HSBC":
        for index, row in df.iterrows():
            word_list = ["apr", "may", "jan","feb","mar","jun","jul","aug","sep","oct","nov","dec",'0ct']
            my_string = str(row[dates])

            found_any = any(word in my_string.lower() for word in word_list)
            if not 'forward' in row[decname].lower():
                if found_any:
                    if not(pd.isna(row[debitname]) or row[debitname] ==''):
                        final_df.loc[len(final_df)]  = [row[dates],row[decname],'Debit',row[debitname],row[balance]]
                        # df.loc[len(df)] 
                        # final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)
                        # #final_df.loc[len(final_df)] = new_row
                    elif not(pd.isna(row[creditname]) or row[creditname] ==''):
                        final_df.loc[len(final_df)]  = [row[dates],row[decname],'Credit',row[creditname],row[balance]]
                        # final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)

    # #2317
    if templates=="U.S. BANK":
        df[creditname] = df[creditname].astype(str)
        for index, row in df.iterrows():
        
            # try:
            #     date_string = "2024-12-25"
            #     format_string = "%Y-%m-%d"
            #     date_object = datetime.strptime(row[dates].str.split('\n')[0] + ' ' + str(years), format_string).date()
            #     print(f"Successfully converted to a date object: {date_object}")
            #     print(f"Type: {type(date_object)}")
                
            # except ValueError as e:
            #     print(f"Error: {e}")
            word_list = ["apr", "may", "jan","feb","mar","jun","jul","aug","sep","oct","nov","dec"]
            word_list1 = ["total"]
            my_string = str(row[dates])
            my_string1 = str(row[decname] + ' ' + row[creditname])


            found_any = any(word in my_string.lower() for word in word_list)
            found_any1 = any(word in my_string1.lower() for word in word_list1)
            #print(found_any1)
            if found_any and not found_any1:
                s1 = str(row[dates]).replace('Mobile Banking Transfer','').replace('\n','').replace('Electronic Deposit','')
                if (is_number(str(row[balance]).replace(',','').replace('—','').replace('-',''))):
                    if not(pd.isna(row[balance]) or row[balance] ==''):
                        if '—' in str(row[balance]) or '-' in str(row[balance]):
                            new_row = [s1 + ' ' + str(years),my_string1,'Debit',row[balance]]
                            final_df.loc[len(final_df)]  = [s1 + ' ' + str(years),row[decname] + ' ' + row[creditname],'Debit',row[balance].replace('—','').replace('-',''),'']
                        else:
                            new_row = [s1 + ' ' + str(years),my_string1,'Credit',row[balance]]
                            final_df.loc[len(final_df)]  = [s1 + ' ' + str(years),row[decname] + ' ' + row[creditname],'Credit',row[balance],'']
                        
                        #final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)
                    #final_df.loc[len(final_df)] = new_row
                
    # #1725
    # debitname = df.columns[8]
    # balance = df.columns[4]
    # creditname = df.columns[3]
    # decname = df.columns[5]
    # dates = df.columns[0]
    # df[decname] = df[decname].astype(str)
    # df[creditname] = df[creditname].astype(str)
    # for index, row in df.iterrows():
        
    #     # try:
    #     #     date_string = "2024-12-25"
    #     #     format_string = "%Y-%m-%d"
    #     #     date_object = datetime.strptime(row[dates].str.split('\n')[0] + ' ' + str(years), format_string).date()
    #     #     print(f"Successfully converted to a date object: {date_object}")
    #     #     print(f"Type: {type(date_object)}")
            
    #     # except ValueError as e:
    #     #     print(f"Error: {e}")
    #     word_list = ["apr", "may", "jan","feb","mar","jun","jul","aug","sep","oct","nov","dec"]
    #     my_string = str(row[dates])
    #     print(str(row[debitname]))

    #     s1 = str(row[dates]).replace('Mobile Banking Transfer','').replace('\n','').replace('Electronic Deposit','')
    #     if not(pd.isna(row[debitname]) or row[debitname] ==''):
    #         if '—' in str(row[debitname]):
    #             new_row = [s1 ,row[decname] + ' ' + row[creditname],'Debit',row[debitname]]
    #             final_df.loc[len(final_df)]  = [row[dates],row[decname],'Debit',row[debitname],'']
    #         else:
    #             new_row = [s1,row[decname] + ' ' + row[creditname],'Credit',row[debitname]]
    #             final_df.loc[len(final_df)]  = [row[dates],row[decname],'Credit',row[debitname],'']
            
            #final_df.loc[len(final_df)] = new_row

    # columns_to_check = [df.columns[2],df.columns[3]]
    # df.loc[df[df.columns[1]].str.contains('forward',case=False,na=False), df.columns[3]] = np.nan
    # df.loc[df[df.columns[2]].notna(), df.columns[3]] = np.nan
    # mask = df[columns_to_check].notna().all(axis=1)
    # df_non_empty = df[mask]
    print(final_df)
    st.subheader("Extracted Transactions")
    st.dataframe(final_df)

    final_df.to_csv(fl + '1.csv',encoding='utf-8-sig')
    final_df.to_json(fl + '1.json',orient='records',indent=4)