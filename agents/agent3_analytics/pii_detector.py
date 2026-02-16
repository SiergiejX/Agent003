"""
PII Detector for Agent3 Analytics
Detects and warns about Personal Identifiable Information in queries.
"""
import re
from typing import List, Dict, Any


class PIIDetector:
    """Detect PII (Personal Identifiable Information) in text."""
    
    def __init__(self):
        # Regex patterns for PII detection
        self.patterns = {
            "index": {
                "pattern": r"\b\d{5}\b",
                "description": "Numer indeksu studenta (5 cyfr)",
                "example": "12345"
            },
            "email": {
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "description": "Adres email",
                "example": "jan.kowalski@example.com"
            },
            "pesel": {
                "pattern": r"\b\d{11}\b",
                "description": "Numer PESEL (11 cyfr)",
                "example": "92032912345",
                "validator": self._validate_pesel
            }
        }
    
    def _validate_pesel(self, pesel_str: str) -> bool:
        """
        Validate PESEL number using checksum algorithm.
        PESEL format: YYMMDDSSSSQ where Q is checksum digit
        """
        if len(pesel_str) != 11 or not pesel_str.isdigit():
            return False
        
        # PESEL weights for checksum
        weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
        
        try:
            # Calculate checksum
            checksum = 0
            for i in range(10):
                checksum += int(pesel_str[i]) * weights[i]
            
            checksum = checksum % 10
            checksum = (10 - checksum) % 10
            
            # Validate checksum digit
            return checksum == int(pesel_str[10])
        except (ValueError, IndexError):
            return False
    
    def detect_pii(self, text: str) -> Dict[str, Any]:
        """
        Detect PII in text.
        
        Returns:
            Dict with:
                - has_pii: bool
                - detections: List of detected PII items with type and value
                - warning_message: str
        """
        detections = []
        
        for pii_type, config in self.patterns.items():
            pattern = config["pattern"]
            matches = re.finditer(pattern, text)
            
            for match in matches:
                value = match.group()
                
                # Additional validation for PESEL
                if pii_type == "pesel":
                    if not self._validate_pesel(value):
                        continue  # Skip if not a valid PESEL
                
                # For index, skip if it's clearly part of a date or other context
                if pii_type == "index":
                    # Check if it's part of a date (e.g., 2023-01-15)
                    context_start = max(0, match.start() - 5)
                    context_end = min(len(text), match.end() + 5)
                    context = text[context_start:context_end]
                    
                    # Skip if part of date pattern or year
                    if re.search(r'\d{4}[-./]\d{2}[-./]\d{2}', context):
                        continue
                    if re.search(r'(19|20)\d{2}', value):  # Year pattern
                        continue
                
                detections.append({
                    "type": pii_type,
                    "value": value,
                    "description": config["description"],
                    "position": match.start()
                })
        
        # Generate warning message if PII detected
        warning_message = ""
        if detections:
            warning_message = self._generate_warning(detections)
        
        return {
            "has_pii": len(detections) > 0,
            "detections": detections,
            "warning_message": warning_message
        }
    
    def _generate_warning(self, detections: List[Dict[str, Any]]) -> str:
        """Generate user-friendly warning message."""
        warning = "⚠️ OSTRZEŻENIE: Wykryto dane wrażliwe w zapytaniu!\n\n"
        warning += "Znaleziono następujące dane osobowe (PII):\n"
        
        for detection in detections:
            pii_type = detection["type"]
            description = detection["description"]
            value_masked = self._mask_value(detection["value"], pii_type)
            
            warning += f"  • {description}: {value_masked}\n"
        
        warning += "\n⛔ UWAGA RODO:\n"
        warning += "System Agent3 Analytics nie powinien przetwarzać danych osobowych.\n"
        warning += "Proszę usunąć dane wrażliwe z zapytania i użyć:\n"
        warning += "  - Zamiast numeru indeksu: 'student grupa A' lub 'studenci 3 roku'\n"
        warning += "  - Zamiast PESEL: 'studenci urodzeni w 1995' lub 'rocznik 1995'\n"
        warning += "  - Zamiast email: 'student informatyki' lub 'studenci wydziału X'\n"
        warning += "\nDane w systemie są zanonimizowane i używają hashów.\n"
        warning += "Zapytanie zostało ZABLOKOWANE ze względów bezpieczeństwa."
        
        return warning
    
    def _mask_value(self, value: str, pii_type: str) -> str:
        """Mask PII value for display."""
        if pii_type == "pesel":
            # Show only first 2 digits (birth year) and mask rest
            return f"{value[:2]}*********"
        elif pii_type == "index":
            return f"{value[0]}****"
        elif pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                masked_username = username[0] + "*" * (len(username) - 1)
                return f"{masked_username}@{domain}"
        return "****"
    
    def sanitize_text(self, text: str) -> str:
        """
        Remove detected PII from text (for logging/storage).
        Replaces:
        - index numbers -> [INDEX]
        - PESEL numbers -> [PESEL]
        - email addresses -> [MAIL]
        """
        sanitized = text
        detection_result = self.detect_pii(text)
        
        if detection_result["has_pii"]:
            # Sort by position in reverse to avoid index shifting
            detections = sorted(detection_result["detections"], 
                              key=lambda x: x["position"], 
                              reverse=True)
            
            for detection in detections:
                value = detection["value"]
                pii_type = detection["type"]
                
                # Use short labels as requested
                label_map = {
                    "index": "[INDEX]",
                    "pesel": "[PESEL]",
                    "email": "[MAIL]"
                }
                replacement = label_map.get(pii_type, f"[{pii_type.upper()}]")
                sanitized = sanitized.replace(value, replacement)
        
        return sanitized


# Test function
def test_pii_detector():
    """Test PII detector with various inputs."""
    detector = PIIDetector()
    
    test_cases = [
        "Jaki jest GPA studenta 12345?",  # Index
        "Student o PESEL 92032912345 ma problem",  # Valid PESEL
        "Student 12345678901 ma pytanie",  # Invalid PESEL (wrong checksum)
        "Kontakt: jan.kowalski@example.com",  # Email
        "Retencja w roku 2023-01-15 wynosiła 78%",  # Date, not index
        "Analiza dla roku 2024",  # Year, not index
        "Student 12345 (jan.kowalski@uczelnia.pl) PESEL 44051401458",  # Multiple PII
        "Jaka jest retencja na informatyce?",  # No PII
    ]
    
    print("="*80)
    print("PII DETECTOR TEST")
    print("="*80 + "\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {text}")
        result = detector.detect_pii(text)
        
        if result["has_pii"]:
            print(f"❌ PII DETECTED:")
            for detection in result["detections"]:
                print(f"   - {detection['description']}: {detector._mask_value(detection['value'], detection['type'])}")
            print(f"\nSanitized: {detector.sanitize_text(text)}")
        else:
            print(f"✅ No PII detected")
        
        print()


if __name__ == "__main__":
    test_pii_detector()
