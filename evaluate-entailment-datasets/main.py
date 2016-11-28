#!/usr/bin/python3
# -*- coding:utf-8 -*-

from util import LEDSDataLoader as leds
from util import BLESSDataLoader as bless
from util import TM14DataLoader as tm14
from util import MedicalDataLoader as medical
from Dataset import Dataset as ds

# LEDSData = ds(leds().load('../datasets/LEDS/positive-examples.txtinput'))
# LEDSData.test('LEDS_result')

# BLESSData = ds(bless().load('../datasets/BLESS/BLESS.txt'))
# BLESSData.test('BLESS_result')

# TM14Data = ds(tm14().load('../datasets/TM14/tm14.tsv'))
# TM14Data.test('TM14_result')

MedicalData = ds(medical().load('../datasets/Medical/medical.tsv'))
MedicalData.test('Medical_result')
