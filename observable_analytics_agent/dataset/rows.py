"""Row types of the five tables, with fields in the same order as the table columns."""

from datetime import date, datetime
from decimal import Decimal
from typing import NamedTuple


class Customer(NamedTuple):
    customer_id: int
    full_name: str
    email: str
    country: str
    segment: str
    signup_date: date
    account_notes: str | None


class Product(NamedTuple):
    product_id: int
    product_name: str
    category: str
    unit_price: Decimal
    launched_on: date
    description: str


class Order(NamedTuple):
    order_id: int
    customer_id: int
    ordered_at: datetime
    status: str
    channel: str


class OrderItem(NamedTuple):
    order_item_id: int
    order_id: int
    product_id: int
    quantity: int
    unit_price: Decimal


class Ticket(NamedTuple):
    ticket_id: int
    customer_id: int
    order_id: int | None
    opened_at: datetime
    resolved_at: datetime | None
    category: str
    priority: str
    status: str
    subject: str
    body: str


ROW_TYPES = {
    "customers": Customer,
    "products": Product,
    "orders": Order,
    "order_items": OrderItem,
    "support_tickets": Ticket,
}
