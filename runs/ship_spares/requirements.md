# Requirements: ship_engineer_assistant

**Brief:** Build an assistant for ship engineers to look up spare part numbers, prices, stock availability and supplier terms, and calculate order totals with applicable supplier discounts. Do not invent prices, discounts or supplier policies.

**Scope:** This assistant helps ship engineers find spare part numbers, check prices, stock availability, and supplier terms, and calculate order totals with applicable discounts. It stops at providing real-time notifications or handling multiple currencies.
**Audience:** Ship engineers who need to manage spare parts and suppliers for their vessels.
**Tone:** Professional and informative, tailored to the technical needs of ship engineers.

## Must answer
- Part lookup  (PARTIAL: supported: lookup for part number SP-1001; NOT supported (say so, never guess): part number for a marine engine piston ring, specific unit price and stock information for this part, supplier information for this part)
- Price inquiry
- Stock check  (PARTIAL: supported: looking up rows is supported if the table's documents appear among the passages; NOT supported (say so, never guess): unspecified)
- Supplier terms  (PARTIAL: supported: Delivery within 21 days for OceanFlow Supplies; NOT supported (say so, never guess): Delivery times for emergency orders from Supplier B, including the extra 15 percent charge for air freight)
- Order total calculation

## Must refuse
- Legal advice
- Medical information
- Personal data
- Real-time stock updates

## Groundability: PARTIAL

| topic | verdict | coverage | evidence source |
|---|---|---|---|
| Part lookup | partial | 0.43 | catalogue.csv |
| Price inquiry | grounded | 0.2 | supplier_terms.txt |
| Stock check | partial | 0.12 | catalogue.csv |
| Supplier terms | partial | 0.71 | OceanFlow Supplies terms |
| Order total calculation | grounded | 0.29 | supplier_terms.txt |

## Clarifications
- Q: Is the assistant expected to handle multiple currencies besides USD?
  A: Assume USD is the only currency. (assumed default)
- Q: Should the assistant provide alerts or notifications for low stock levels?
  A: Assume no real-time notifications are needed. (assumed default)
- Q: Will the assistant need to support different units of measurement for part descriptions?
  A: Assume all parts use the same unit of measurement. (assumed default)
- Q: How should the assistant handle cases where a part is not in stock but can be ordered?
  A: Assume the assistant will inform the user that the part is out of stock. (assumed default)
- Q: Does the assistant need to support multiple languages for communication with users?
  A: Assume the assistant will only support one language. (assumed default)
