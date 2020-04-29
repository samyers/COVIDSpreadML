import numpy as np
import pandas as pd
from datetime import datetime as dt
import csv


def load_county_cases_data(filename, use_full_year, valid_counties):
  with open(filename, 'r', errors='ignore') as fin:
    df = pd.read_csv(fin)
    
    df = df[df['countyFIPS'] > 1]
    first_date_column = '1/22/20'
    if use_full_year:
      first_date_column += '20'
    
    cases = df.loc[:, first_date_column:].to_numpy()
    dates_str = df.columns.to_list()[df.columns.get_loc(first_date_column):]
    dates = [dt.strptime(d, '%m/%d/%y' if not use_full_year else '%m/%d/%Y').date() for d in dates_str]
    df = pd.DataFrame(cases, index=df['countyFIPS'], columns=dates)
    for r in df.index:
      if r not in valid_counties:
        df = df.drop(r)
    return df


def get_state2county(filename):
  state2county = {}
  with open(filename, 'r', errors='ignore') as fin_raw:
    fin = csv.reader(fin_raw, dialect=csv.excel)
    _ = fin.__next__()
    for line_ndx in fin:
      if line_ndx[0] != '0':
        state = line_ndx[2]
        if state not in state2county:
          state2county[state] = []
        state2county[state].append(int(line_ndx[0]))
    return state2county


def get_county_info(filename):
  county_info = {}
  with open(filename, 'r', errors='ignore') as fin_raw:
    fin = csv.reader(fin_raw, dialect=csv.excel)
    headers = fin.__next__()
    header_ndx = {h: i for i, h in enumerate(headers)}
    
    for line_ndx in fin:
      if line_ndx[0] != '0':
        county_info[line_ndx[0]] = {}
        county_info[line_ndx[0]]['County Name'] = line_ndx[header_ndx['County Name']]
        county_info[line_ndx[0]]['State'] = line_ndx[header_ndx['State']]
    return county_info


def load_test_data_crv(filename, all_dates, all_counties, state2county, cases_by_county):
  with open(filename, 'r', errors='ignore') as fin_raw:
    positive_tests = {c: {d: -1 for d in all_dates} for c in all_counties}
    negative_tests = {c: {d: -1 for d in all_dates} for c in all_counties}
    hospitals = {c: {d: -1 for d in all_dates} for c in all_counties}
    fin = csv.reader(fin_raw, dialect=csv.excel)
    headers = fin.__next__()
    header_ndx = {h: i for i, h in enumerate(headers)}

    for line_ndx in fin:
      date = dt.strptime(line_ndx[header_ndx['date']], '%Y%m%d').date()
      state = line_ndx[header_ndx['state']]
      pos = float(line_ndx[header_ndx['positive']].replace(',', '') or 0)
      neg = float(line_ndx[header_ndx['negative']].replace(',', '') or 0)
      hosp = float(line_ndx[header_ndx['hospitalized']].replace(',', '') or 0)
      if state in state2county:
        if date in all_dates:
          for county in state2county[state]:
            if county in all_counties:
              positive_tests[county][date] = pos
              negative_tests[county][date] = neg
              hospitals[county][date] = hosp

    positive_tests = pd.DataFrame.from_dict(positive_tests, orient='index', columns=all_dates)
    positive_tests.fillna(-1, inplace=True)

    negative_tests = pd.DataFrame.from_dict(negative_tests, orient='index', columns=all_dates)
    negative_tests.fillna(-1, inplace=True)

    hospitals = pd.DataFrame.from_dict(hospitals, orient='index', columns=all_dates)
    hospitals.fillna(-1, inplace=True)

    return positive_tests, negative_tests, hospitals


def load_unacast_data_crv(filename, all_dates, all_counties):

  with open(filename, 'r', errors='ignore') as fin_raw:
    unacast_distance_dff = {c: {d: -1 for d in all_dates} for c in all_counties}
    unacast_visitation_dff = {c: {d: -1 for d in all_dates} for c in all_counties}
    unacast_total = {c: {d: -1 for d in all_dates} for c in all_counties}
    fin = csv.reader(fin_raw, dialect=csv.excel)
    headers = fin.__next__()
    header_ndx = {h: i for i, h in enumerate(headers)}
    
    for line_ndx in fin:
      date = dt.strptime(line_ndx[header_ndx['date']], '%Y-%m-%d').date()
      county = int(line_ndx[header_ndx['county_fips']])
      
      total = float(line_ndx[header_ndx['n_grade_total']].replace(',', '') or 0)
      dd = float(line_ndx[header_ndx['n_grade_distance']].replace(',', '') or 0)
      vd = float(line_ndx[header_ndx['n_grade_visitation']].replace(',', '') or 0)
      if date in all_dates and county in all_counties:
        unacast_distance_dff[county][date] = dd
        unacast_visitation_dff[county][date] = vd
        unacast_total[county][date] = total

    unacast_total = pd.DataFrame.from_dict(unacast_total, orient='index', columns=all_dates)
    unacast_total.fillna(-1, inplace=True)
    
    unacast_distance_dff = pd.DataFrame.from_dict(unacast_distance_dff, orient='index', columns=all_dates)
    unacast_distance_dff.fillna(-1, inplace=True)
    
    unacast_visitation_dff = pd.DataFrame.from_dict(unacast_visitation_dff, orient='index', columns=all_dates)
    unacast_visitation_dff.fillna(-1, inplace=True)

    return unacast_distance_dff, unacast_visitation_dff, unacast_total


def load_distance_traveled_crv(filename, all_dates, all_counties):
  with open(filename, 'r', errors='ignore') as fin_raw:
    fin = csv.reader(fin_raw, dialect=csv.excel)
    headers = fin.__next__()
    header_ndx = {h: i for i, h in enumerate(headers)}
    
    parsed_county_ids = {}
    df_data = {c: {d: -1 for d in all_dates} for c in all_counties}
    
    county_info = load_census_data('data/population_densities.csv',
                                   all_counties=all_counties)
    
    for line_ndx in fin:
      date = dt.strptime(line_ndx[header_ndx['ref_dt']], '%Y-%m-%d').date()
      county_str = line_ndx[header_ndx['countyfp10']]
      state_str = line_ndx[header_ndx['statefp10']]
      if (county_str, state_str) not in parsed_county_ids:
        while len(county_str) < 3:
          county_str = '0' + county_str
        county = int(state_str + county_str)
        parsed_county_ids[(county_str, state_str)] = county
      
      county = parsed_county_ids[(county_str, state_str)]
      v = float(line_ndx[header_ndx['county_vmt']]) / county_info.loc[county, 'Area']
      df_data[county][date] = v
    
    df = pd.DataFrame.from_dict(df_data, orient='index', columns=all_dates)
    df.fillna(-1, inplace=True)
    
    return df


def load_census_data(filename, all_counties):
  all_counties_set = set(all_counties.to_list())
  with open(filename, 'r', errors='ignore') as fin_raw:
    fin = csv.reader(fin_raw, dialect=csv.excel)
    df = pd.DataFrame(data=0.0, index=all_counties, columns=['Population', 'Density', 'Per Household', 'Area'])
    fin.__next__()
    for line_ndx in fin:
      county_id = int(line_ndx[4] or -1)
      if all_counties_set is None or county_id in all_counties_set:
        df.at[county_id, 'Population'] = int(line_ndx[7].split('(r')[0])
        df.at[county_id, 'Density'] = float(line_ndx[12].split('(r')[0])
        df.at[county_id, 'Per Household'] = float(line_ndx[7].split('(r')[0]) / (1.0 + float(line_ndx[8].split('(r')[0]))
        df.at[county_id, 'Area'] = float(line_ndx[11])
    
    return df


def get_all_counties(filename):
  all_counties = []
  with open(filename, 'r', errors='ignore') as fin_raw:
    fin = csv.reader(fin_raw, dialect=csv.excel)
    fin.__next__()
    for line_ndx in fin:
      all_counties.append(int(line_ndx[4] or -1))
    return all_counties


def fill_forward(df):
  """
  For time-based data frames, fill in any missing days of data for a county using the most recent known value.
  
  Note: the performance for this method is pretty bad, could stand to be optimized.
  :param df:
  :return:
  """
  d = dt(2020, 1, 1).date()
  for c in df.columns:
    if isinstance(c, d.__class__):
      for c2 in df.columns[df.columns > c]:
        df_c = df.loc[:, c]
        df_c2 = df.loc[:, c2]
        df.at[df_c > df_c2, c2] = df_c.loc[df_c > df_c2]
  return df


def get_all_datasets():
  """
  Loads the various datasets from local disk and returns them as Pandas DataFrames. For the time-based datasets, the rows
  are counties (the row index being the county ID, commonly referred to as countyFIPS), and the columns dates
   For non-time based datasets, the rows are counties and the columns are variable names
  :return:
       confirmed_cases_df: the number of covid cases per county over time
       deaths_df: the number of covid-related deaths per county over time
       positive_tests_df: the number of positive tests state-wide for each county over time
       negative_tests_df: the number of negative tests state-wide for each county over time
       hospitals_df: the number of hospitalizations state-wide for each county over time
       census_df: population and population density data per county (not time-based)
       unacast_distance_dff: social distancing scores provided by Unacast per county over time
       unacast_visitation_dff: more social distancing scores provided by Unacast per county over time
       unacast_total: even more social distancing scores provided by Unacast per county over time
       distance_traveled: total miles traveled per county each day
       education_df: education distribution per county (not time based)
       race_df: racial distribution per county (not time based)
       employment_df: employment rate and income per county (not time based)
  """
  print('loading confirmed case counts')
  #some of the data sources include non-county data. Use the US Census data to get the valid county ids
  all_counties = get_all_counties('data/population_densities.csv')
  confirmed_cases_df_init = load_county_cases_data('data/covid_confirmed_cases.csv', False, all_counties)
  
  confirmed_cases_df = confirmed_cases_df_init
  
  confirmed_cases_df = fill_forward(confirmed_cases_df)
  
  print('loading death counts')
  deaths_df = load_county_cases_data('data/covid_deaths.csv', False, all_counties)
  
  state2county = get_state2county('data/covid_confirmed_cases.csv')
  
  print('loading testing data')
  positive_tests_df, negative_tests_df, hospitals_df = load_test_data_crv('data/testing_hospitals_deaths.csv',
                                                                          all_dates=confirmed_cases_df_init.columns,
                                                                          all_counties=confirmed_cases_df_init.index,
                                                                          state2county=state2county,
                                                                          cases_by_county=confirmed_cases_df)

  positive_tests_df = fill_forward(positive_tests_df)
  negative_tests_df = fill_forward(negative_tests_df)
  hospitals_df = fill_forward(hospitals_df)

  print('loading demographic data')
  census_df = load_census_data('data/population_densities.csv',
                               all_counties=confirmed_cases_df_init.index)
  
  education_in = open('data/county_education.csv', 'r')
  education_df = pd.read_csv(education_in, index_col=0)
  education_in.close()
  
  race_in = open('data/county_race.csv', 'r')
  race_df = pd.read_csv(race_in, index_col=0)
  race_in.close()
  
  employment_in = open('data/county_income_employment.csv', 'r')
  employment_df = pd.read_csv(employment_in, index_col=0)
  employment_in.close()
  
  print('loading mobility data')
  unacast_distance_dff, unacast_visitation_dff, unacast_total = load_unacast_data_crv(
    filename='data/0401_sds-full-county.csv',
    all_dates=confirmed_cases_df_init.columns,
    all_counties=confirmed_cases_df_init.index,
  )
  
  
  unacast_distance_dff = fill_forward(unacast_distance_dff)
  unacast_visitation_dff = fill_forward(unacast_visitation_dff)
  unacast_total = fill_forward(unacast_total)
  
  distance_traveled = load_distance_traveled_crv(
    filename='data/distance_traveled.csv',
    all_dates=confirmed_cases_df_init.columns,
    all_counties=confirmed_cases_df_init.index,
  )

  distance_traveled = fill_forward(distance_traveled)
  
  return confirmed_cases_df, \
         deaths_df, \
         positive_tests_df, \
         negative_tests_df, \
         hospitals_df, \
         census_df, \
         unacast_distance_dff, \
         unacast_visitation_dff, \
         unacast_total, \
         distance_traveled, \
         education_df, \
         race_df, \
         employment_df


