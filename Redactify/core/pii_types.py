#!/usr/bin/env python3
# Redactify/core/pii_types.py

"""
Centralized PII type definitions for Redactify.
This module contains all PII type definitions organized by categories.
"""

from typing import Dict, List, Tuple

# --- PII Type Categories ---

class PiiCategory:
    """A class to represent a category of PII types with related functionality."""
    
    def __init__(self, name: str, description: str, pii_types: List[Tuple[str, str]]):
        """
        Initialize a PII category
        
        Args:
            name: The name of the category
            description: A description of the category
            pii_types: List of (entity_id, friendly_name) tuples
        """
        self.name = name
        self.description = description
        self.pii_types = pii_types
        
    def get_entity_ids(self) -> List[str]:
        """Returns a list of entity IDs in this category."""
        return [entity_id for entity_id, _ in self.pii_types]
    
    def get_friendly_names(self) -> List[Tuple[str, str]]:
        """Returns the list of (entity_id, friendly_name) tuples."""
        return self.pii_types


# --- PII Category Definitions ---

# Common PII types (India-specific + basic universal types)
COMMON = PiiCategory(
    name="common",
    description="Commonly used PII types for basic redaction needs",
    pii_types=[
        ('PERSON', 'Person Name'),
        ('PHONE_NUMBER', 'Phone Number'),
        ('EMAIL_ADDRESS', 'Email Address'),
        ('LOCATION', 'Address & Location'),
        ('INDIA_AADHAAR_NUMBER', 'Aadhaar Number'),
        ('INDIA_PAN_NUMBER', 'PAN Card Number'),
        ('INDIA_PASSPORT', 'Indian Passport Number'),
        ('INDIA_VOTER_ID', 'Voter ID (EPIC Number)'),
    ]
)

# Advanced PII types (less common or international types)
ADVANCED = PiiCategory(
    name="advanced",
    description="Specialized PII types for specific regulatory or industry compliance",
    pii_types=[
        ('CREDIT_CARD', 'Credit Card Number'),
        ('BANK_ACCOUNT', 'Bank Account Number'),
        ('US_SSN', 'US Social Security Number'),
        ('DATE_TIME', 'Date & Time'),
        ('NRP', 'National Registration/ID Number'),
        ('URL', 'Website URL'),
        ('IBAN_CODE', 'International Bank Account Number'),
        ('IP_ADDRESS', 'IP Address'),
        ('MEDICAL_LICENSE', 'Medical License Number'),
        ('EXAM_IDENTIFIER', 'Exam IDs & Roll Numbers'),
        ('DRIVING_LICENSE', 'Driving License'),
        ('US_DRIVER_LICENSE', 'US Driver License'),
        ('US_ITIN', 'US Individual Taxpayer ID'),
        ('US_PASSPORT', 'US Passport Number'),
        ('UK_NHS', 'UK NHS Number'),
        ('UK_PASSPORT', 'UK Passport Number'),
        ('SWIFT_CODE', 'SWIFT Code'),
    ]
)

# Financial PII types (could be expanded in future)
FINANCIAL = PiiCategory(
    name="financial",
    description="Financial and banking related PII types",
    pii_types=[
        ('CREDIT_CARD', 'Credit Card Number'),
        ('BANK_ACCOUNT', 'Bank Account Number'),
        ('IBAN_CODE', 'International Bank Account Number'),
        ('SWIFT_CODE', 'SWIFT Code'),
    ]
)

# --- Registry of all categories ---
CATEGORIES: Dict[str, PiiCategory] = {
    "common": COMMON,
    "advanced": ADVANCED,
    "financial": FINANCIAL,
    # Add new categories here as needed
}

# --- Helper Functions ---

def get_all_pii_types() -> List[Tuple[str, str]]:
    """Returns a combined list of all PII types from all categories."""
    all_types = []
    # Use a set to track added types to avoid duplicates
    added_types = set()
    
    for category in CATEGORIES.values():
        for entity_id, friendly_name in category.pii_types:
            if entity_id not in added_types:
                all_types.append((entity_id, friendly_name))
                added_types.add(entity_id)
                
    return all_types

def get_pii_types(category_name: str = None, advanced: bool = None) -> List[str]:
    """
    Returns list of PII entity type IDs for a given category.
    
    Args:
        category_name: Name of the category to retrieve types from
        advanced: (Backward compatibility) If True, returns advanced types, if False returns common types
        
    Returns:
        list: List of PII entity type IDs
    """
    # Handle backward compatibility with advanced parameter
    if advanced is not None:
        category_name = "advanced" if advanced else "common"
    elif category_name is None:
        # Default to common if neither parameter is specified
        category_name = "common"
        
    if category_name not in CATEGORIES:
        raise ValueError(f"Unknown PII category: {category_name}")
        
    return CATEGORIES[category_name].get_entity_ids()

def get_pii_friendly_names(category_name: str = None, advanced: bool = None) -> List[Tuple[str, str]]:
    """
    Returns the mapping of PII entity types to their user-friendly names for a given category.
    
    Args:
        category_name: Name of the category to retrieve friendly names from
        advanced: (Backward compatibility) If True, returns advanced types, if False returns common types
        
    Returns:
        list: List of (entity_type, friendly_name) tuples
    """
    # Handle backward compatibility with advanced parameter
    if advanced is not None:
        category_name = "advanced" if advanced else "common"
    elif category_name is None:
        # Default to common if neither parameter is specified
        category_name = "common"
        
    if category_name not in CATEGORIES:
        raise ValueError(f"Unknown PII category: {category_name}")
        
    return CATEGORIES[category_name].get_friendly_names()

# Legacy compatibility functions
def get_common_pii_types() -> List[str]:
    """Returns list of common PII entity type IDs. For backwards compatibility."""
    return get_pii_types("common")

def get_advanced_pii_types() -> List[str]:
    """Returns list of advanced PII entity type IDs. For backwards compatibility."""
    return get_pii_types("advanced")