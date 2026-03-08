// Arrow schema for COTAHIST parsed records

use arrow::datatypes::{DataType, Field, Schema};

/// Returns the Arrow schema for COTAHIST parsed records.
///
/// Columns (in order):
///   date             - Utf8 "YYYY-MM-DD" (not nullable)
///   ticker           - Utf8 (not nullable)
///   isin_code        - Utf8 (not nullable)
///   open             - Float64 (not nullable)
///   high             - Float64 (not nullable)
///   low              - Float64 (not nullable)
///   close            - Float64 (not nullable)
///   volume           - Float64 (not nullable)
///   quotation_factor - Int64 (not nullable)
pub fn cotahist_schema() -> Schema {
    Schema::new(vec![
        Field::new("date", DataType::Utf8, false),
        Field::new("ticker", DataType::Utf8, false),
        Field::new("isin_code", DataType::Utf8, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
        Field::new("quotation_factor", DataType::Int64, false),
    ])
}
