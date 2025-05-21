import PySimpleGUI as sg
from src.training import run_all


def launch_gui():
    sg.theme('LightBlue')
    layout = [
        [sg.Text('Raw Data Folder'), sg.InputText(r'C:\Dev\Python\Position_Estimation_Code\data\raw', key='-RAW-'), sg.FolderBrowse()],
        [sg.Text('Processed Data Folder'), sg.InputText(r'C:\Dev\Python\Position_Estimation_Code\data\processed', key='-PROCESSED-'), sg.FolderBrowse()],
        [sg.Text('Decimals for R'), sg.InputText('5', key='-DEC-')],
        [sg.Button('Run'), sg.Button('Exit')],
        [sg.Output(size=(80, 20))]
    ]

    window = sg.Window('Position Estimation RMSE App', layout)
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'Run':
            raw_path = values['-RAW-']
            proc_path = values['-PROCESSED-']
            decimals = int(values['-DEC-'])
            print(f"Processing: R decimals={decimals}")
            run_all(raw_path, proc_path, decimals)
    window.close()