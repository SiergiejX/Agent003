# Agent3_analytics – Analiza danych i trendów (BOS)

Repozytorium dokumentuje agenta **agent3_analytics** – analityczną część wieloagentowego systemu wsparcia **Biura Obsługi Studenta (BOS)** (IBM watsonx + Qdrant).

> Agent3 jest agentem typu **RAG**: odpowiada na zapytania analityczne na podstawie danych i kolekcji w wektorowej bazie danych **Qdrant**.

## Cel projektu
- Umożliwienie pracownikom uczelni uzyskiwania analiz i trendów poprzez zapytania w języku naturalnym.
- Agregacja danych z wielu źródeł (pozostałych agentów oraz kolekcji nieagentowych).
- Zapewnienie zgodności i sanityzacji zapytań przez warstwę bezpieczeństwa (agent5).

## Agenci w systemie BOS
- **agent1** – chatbot dla studentów (rozmowy + dokumenty uczelni)
- **agent2** – system ticketowy i klasyfikacja zgłoszeń
- **agent3** – analiza danych o studentach i trendów (ten projekt)
- **agent4** – generatywne wsparcie dla pracowników BOS
- **agent5** – bezpieczeństwo, zgodność, zarządzanie danymi i anonimizacja

## Wejście / wyjście
- **Input:** pojedyncze zapytanie tekstowe (prompt) w języku naturalnym.
- **Output:** odpowiedź analityczna (lub odmowa / informacja o braku danych / informacja o wyjściu poza zakres kompetencji).

## Przepływ (BPMN)
1. **Analityk** wpisuje zapytanie.
2. **Filtr** ocenia, czy zapytanie wymaga sanityzacji i/lub zawiera treści niedozwolone.
3. Jeśli potrzeba: zapytanie trafia do **agent5_security** (sanityzacja/odmowa).
4. **agent3_analytics**:
   - sprawdza, czy zapytanie mieści się w kompetencjach,
   - wybiera kolekcje do przeszukania,
   - wykonuje retrieval i agregację,
   - generuje odpowiedź.
5. Zapytanie + odpowiedź są zapisywane w kolekcji historii (pamięć analityczna).

## Źródła wiedzy
Agent3 korzysta z kolekcji Qdrant, które dzielą się na 3 grupy:
- **3agentowe** – aktualizowane przez agent3 (np. historia zapytań).
- **Innoagentowe** – aktualizowane przez inne agenty (np. agent2 tickets).
- **Nieagentowe** – kolekcje z danymi uczelni (np. rekrutacja, egzaminy, stypendia).

> Założenie: rekordy w kolekcjach innoagentowych i nieagentowych są wcześniej zanonimizowane/sanitaryzowane (np. przez agent5).

## Kolekcje Qdrant wykorzystywane przez agent3 (proponowane)
Poniżej opisano docelowe kolekcje dla analiz trendów i danych o studentach.

```text
+------------------------------+------------------------------+------------------------------+------------------------------+
| Kolekcja                     | Przeznaczenie                | Przykładowe pola             | Typowe pytania               |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_student_profiles_snapshot | Snapshoty profili studentów  | student_hash, snapshot_date, | Profil studentów odpadających|
|                              | (segmentacja)                | faculty_id, program_id,      | po 1 semestrze?              |
|                              |                              | status, gpa, ects_completed,  |                              |
|                              |                              | failed_courses_count,        |                              |
|                              |                              | risk_score                   |                              |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_academic_events           | Oś czasu zdarzeń             | event_type, event_date,      | Co poprzedza skreślenie?     |
|                              | akademickich                 | student_hash, faculty_id,    |                              |
|                              |                              | program_id, reason_code      |                              |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_course_performance        | Wyniki w przedmiotach        | course_id, term, attempt_no, | Które przedmioty generują    |
|                              |                              | final_grade, passed, ects     | najwięcej niezaliczeń?       |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_retention_cohorts         | Kohorty i retencja           | entry_year, cohort_size,     | Retencja kierunku X w 3 lata?|
|                              | (agregaty)                   | retention_1_sem,             |                              |
|                              |                              | retention_1_year, dropout_rate|                              |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_support_interactions_sum  | Ticketing (sygnały problemów) | ticket_id_hash, student_hash,| Czy tickety o opłatach       |
|                              |                              | category, priority, resolved,| korelują ze skreśleniami?    |
|                              |                              | resolution_time              |                              |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_policies_and_rules_analyt | Regulaminy / procedury       | doc_type, topic,             | Czy zmiana regulaminu wpływa |
|                              | jako kontekst                | effective_from, version      | na trend urlopów?            |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_reports_and_insights      | Raporty i wnioski agent3      | report_id, period_start,     | Kluczowy problem w semestrze |
|                              |                              | period_end, scope, kpi_tags, | 2024Z?                       |
|                              |                              | confidence                   |                              |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_metrics_timeseries        | Szeregi czasowe KPI          | metric_name, granularity,    | Trend skreśleń miesiąc do    |
|                              |                              | period_from, period_to,      | miesiąca?                    |
|                              |                              | values, units                |                              |
+------------------------------+------------------------------+------------------------------+------------------------------+
| a3_analytics_queries_history | Historia zapytań i odpowiedzi| query_text, intent,          | Czy już analizowaliśmy       |
|                              |                              | filters_used, collections_used,| podobny przypadek?         |
|                              |                              | answer_summary               |                              |
+------------------------------+------------------------------+------------------------------+------------------------------+
text```

## Przykład działania
Zapytanie: **„Ile było zgłoszeń o rezygnacji ze studiów?”**
- Agent3 wybiera kolekcje (np. `a3_academic_events`, `a3_support_interactions_summary`).
- Jeśli brak rekordów: odpowiedź brzmi **„Nie było takiego zgłoszenia.”**

## Konteneryzacja
Projekt jest przewidziany do uruchomienia w kontenerach (szczegóły wdrożenia zależne od orkiestracji całego systemu).

## Promotor
Prof. dr hab. inż. Cezary Orłowski
