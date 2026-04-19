Breaking Procrastination Paralysis

My final year project for BSc in Computer Science (Infrastructure) at TU Dublin.
Andrea Luca - C22390831
Supervisor: Dr. Arthur Sloan

It's a web app that helps people who procrastinate because of overwhelm, anxiety or perfectionism and not because it's a laziness or time management issue. The app uses two ML models to detect how you're feeling and what you're doing about it, then gives you a personalised reframe using your own words and one tiny action to break the freeze.

The app includes:

- Dual ML pipeline (11 emotions, 6 behaviour states)
- 66-entry intervention matrix combining both models
- Three-layer personalised reframes using identity beliefs from onboarding
- Paralysis score (-5 to +5) tracking how stuck you are
- Temporal analysis detecting patterns across same-day entries
- 5-step identity-based onboarding
- Daily, weekly and monthly planning views
- Analytics with Chart.js (mood trends, paralysis score, behaviour distribution)
- Psychology education page with 12 interactive cards
- 6-layer security (CSP, cookies, CSRF, sanitisation, rate limiting, RLS)
- GDPR compliant account deletion
- Custom CSS design system (calm.css)

Built with Python, Flask, scikit-learn, Supabase PostgreSQL. Deployed on Render.
Live at: https://breaking-procrastination-paralysis.onrender.com
