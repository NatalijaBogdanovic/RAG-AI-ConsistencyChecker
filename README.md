ako csv ne prikazuje lepo u linije 17 i 21 u rag.py fajlu zameniti ovim:
csv_bp_writer = csv.writer(csv_bp_file, quoting=csv.QUOTE_ALL, delimiter=';')
csv_p_writer = csv.writer(csv_p_file, quoting=csv.QUOTE_ALL, delimiter=';')

