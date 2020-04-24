import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DataFramePlayground():
    def __init__(self):
        # Load pitcher data into class variable for playing with in this playground
        self._pitcher_data = pd.read_csv('.\\DataFrame\\Pitchers.csv')

    def run(self):
        self._print_data()
        self._modify_data()
        self._sort_and_filter_data()
        self._calc_statistics()
        self._plot_data()

    ###########################################################################
    #  Private Methods
    ###########################################################################

    def _print_data(self):
        # Print leading and trailing data to verify data succesfully loaded
        print('HEAD')
        print(self._pitcher_data.head(n=10))
        print('\n*************************************\n')

        print('TAIL')
        print(self._pitcher_data.tail(n=10))
        print('\n*************************************\n')

        # Print row indices and column headers
        print('INDICES')
        print(self._pitcher_data.index)
        print('\n*************************************\n')

        print('COLUMN NAMES')
        print(self._pitcher_data.columns)
        print('\n*************************************\n')

        # Print the data types
        print('DATA TYPES')
        print(self._pitcher_data.dtypes)
        print('\n*************************************\n')

        # Print the data dimensions
        print('DATA DIMENSIONS')
        print(f'Number of Dimensions: ', {self._pitcher_data.ndim})
        print(f'Number of rows and columns: ', {self._pitcher_data.shape})
        print(f'Number of data values: ', {self._pitcher_data.size})
        print(f'Memory usage:\n')
        print(self._pitcher_data.memory_usage())
        print('\n*************************************\n')

        # Retrieve subset of data via column name in different manners
        print('LAST NAMES')
        last_names = self._pitcher_data['nameLast']
        print(last_names)
        print('\n*************************************\n')

        print('FIRST NAMES')
        first_names = self._pitcher_data.nameFirst
        print(first_names)
        print('\n*************************************\n')

        print('FULL NAME')
        full_name = self._pitcher_data.loc[:, ['nameFirst', 'nameLast']]
        print(full_name)
        print('\n*************************************\n')

        # Retrieve a row
        print('SINGLE ROW')
        row = self._pitcher_data.loc[100] # NOTE: this retrieves the row where ROWID = 100
        print(row)
        print('\n')
        row = self._pitcher_data.iloc[100] # NOTE: this retrieves the row whose index is 100
        print(row)
        print('\n*************************************\n')

        # Convert the data to a NumPy array, removing the column names and row indices
        print('NUMPY ARRAY')
        numpy_array = self._pitcher_data.to_numpy()  # Note: self._pitcher_data.values also works
        print(numpy_array)
        print('\n*************************************\n')

    def _modify_data(self):
        player = pd.Series(data=['turc', 'ToddT'], index=['playerID', 'nameFirst'], name=0)
        self._pitcher_data = self._pitcher_data.append(player)

        # Print new player
        print('NEW PLAYER')
        print(self._pitcher_data.tail(n=10))
        print('\n*************************************\n')

        # Add a new column
        print('NEW COLUMN')
        self._pitcher_data['middleName'] = 'Unspecified'
        print(self._pitcher_data.tail(n=10))
        print('\n*************************************\n')

        # Drop two columns
        print('DROP COLUMNS')
        self._pitcher_data.drop(labels={'retroID', 'bbrefID'}, axis=1, inplace=True)
        print(self._pitcher_data.tail(n=10))
        print('\n*************************************\n')

        # Add a new column and insert values using other columns
        print('ADD NEW COLUMN AND SET VALUE')
        self._pitcher_data['numDecisions'] = self._pitcher_data['W'] + self._pitcher_data['L']
        print(self._pitcher_data.head(n=10))
        print('\n*************************************\n')

    def _sort_and_filter_data(self):
        # Sort by number of wins and losses
        print('WINNINGIEST PITCHERS')
        winningiest_pitchers = self._pitcher_data.sort_values(by=['W', 'L'], ascending=[False, False], inplace=False)
        print(winningiest_pitchers.head(n=10))
        print('\n*************************************\n')

        # Filter by 20 game winners
        print('20 GAME WINNERS')
        won_twenty_games = self._pitcher_data[self._pitcher_data['W'] >= 20]
        print(won_twenty_games.head(n=10))
        print('\n*************************************\n')

    def _calc_statistics(self):
        # Print basic statistics
        print('BASIC STATS')
        print(self._pitcher_data.describe())
        print('\n*************************************\n')

        print(f'Avg Wins: ', {self._pitcher_data['W'].mean(skipna=True)})
        print(f'Min Wins: ', {self._pitcher_data['W'].min(skipna=True)})
        print(f'Max Wins: ', {self._pitcher_data['W'].max(skipna=True)})
        print(f'Std Dev Wins: ', {self._pitcher_data['W'].std(skipna=True)})
        print('\n*************************************\n')

    def _plot_data(self):
        won_twenty_games = self._pitcher_data[self._pitcher_data['W'] >= 20]

        plt.scatter(won_twenty_games['W'], won_twenty_games['G'], alpha=0.5)
        plt.title('Relationship Between 20-game Winners and Number of Appearances')
        plt.xlabel('Wins')
        plt.ylabel('Appearances')
        plt.show()