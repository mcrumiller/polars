use polars_core::prelude::*;

use crate::prelude::*;
use crate::{date_series_from_parts, datetime_series_from_parts};

pub trait PolarsReplaceDatetime {
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
        hour: &Int8Chunked,
        minute: &Int8Chunked,
        second: &Int8Chunked,
        microsecond: &Int32Chunked,
        ambiguous: &StringChunked,
        non_existent: NonExistent,
    ) -> PolarsResult<Column>;
}

pub trait PolarsReplaceDate {
    fn replace(
        &self,
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
        non_existent: NonExistent,
    ) -> PolarsResult<Column>;
}

impl PolarsReplaceDatetime for DatetimeChunked {
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
        hour: &Int8Chunked,
        minute: &Int8Chunked,
        second: &Int8Chunked,
        microsecond: &Int32Chunked,
        ambiguous: &StringChunked,
        non_existent: NonExistent,
    ) -> PolarsResult<Column> {
        let n = self.len();

        // For each argument, we must check if:
        // 1. No value was supplied (None)       --> Use existing year from Series
        // 2. Value was supplied and is a Scalar --> Create full Series of value
        // 3. Value was supplied and is Series   --> Update all elements with the non-null values
        let year = if year.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { year.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    year
                } else {
                    &Int32Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.year()
            }
        } else {
            &year.zip_with(&year.is_not_null(), &self.year())?
        };
        let month = if month.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { month.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    month
                } else {
                    &Int8Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.month()
            }
        } else {
            &month.zip_with(&month.is_not_null(), &self.month())?
        };
        let day = if day.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { day.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    day
                } else {
                    &Int8Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.day()
            }
        } else {
            &day.zip_with(&day.is_not_null(), &self.day())?
        };
        let hour = if hour.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { hour.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    hour
                } else {
                    &Int8Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.hour()
            }
        } else {
            &hour.zip_with(&hour.is_not_null(), &self.hour())?
        };
        let minute = if minute.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { minute.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    minute
                } else {
                    &Int8Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.minute()
            }
        } else {
            &minute.zip_with(&minute.is_not_null(), &self.minute())?
        };
        let second = if second.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { second.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    second
                } else {
                    &Int8Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.second()
            }
        } else {
            &second.zip_with(&second.is_not_null(), &self.second())?
        };
        let microsecond = if microsecond.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { microsecond.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    microsecond
                } else {
                    &Int32Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &(self.nanosecond() / 1000)
            }
        } else {
            &microsecond.zip_with(&microsecond.is_not_null(), &(self.nanosecond() / 1000))?
        };

        let out = datetime_series_from_parts(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            ambiguous,
            &self.time_unit(),
            None,
            &self.name(),
        )?;

        if non_existent == NonExistent::Raise {
            polars_ensure!(
                self.is_null().equal(&self.is_null()).all(),
                ComputeError: "Invalidate datetime component specified.",
            );
        }
        Ok(out)
    }
}

impl PolarsReplaceDate for DateChunked {
    fn replace(
        &self,
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
        non_existent: NonExistent,
    ) -> PolarsResult<Column> {
        let n = self.len();

        let year = if year.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { year.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    year
                } else {
                    &Int32Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.year()
            }
        } else {
            &year.zip_with(&year.is_not_null(), &self.year())?
        };
        let month = if month.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { month.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    month
                } else {
                    &Int8Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.month()
            }
        } else {
            &month.zip_with(&month.is_not_null(), &self.month())?
        };
        let day = if day.len() == 1 {
            // SAFETY: array has one value.
            let value = unsafe { day.get_unchecked(0) };
            if value.is_some() {
                if n == 1 {
                    day
                } else {
                    &Int8Chunked::full("".into(), value.unwrap(), n)
                }
            } else {
                &self.day()
            }
        } else {
            &day.zip_with(&day.is_not_null(), &self.day())?
        };

        let out = date_series_from_parts(year, month, day, &self.name())?;

        if non_existent == NonExistent::Raise {
            polars_ensure!(
                self.is_null().equal(&self.is_null()).all(),
                ComputeError: "Invalidate datetime component specified.",
            );
        }
        Ok(out)
    }
}
