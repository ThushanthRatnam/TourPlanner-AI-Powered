import { useState } from 'react'
import axios from 'axios'
import './App.css'

interface TripPreferences {
  budget: string
  duration: string
  interests: string[]
  startLocation: string
  travelMonth: string
  optimizationLevel: string
}

interface TripPlan {
  itinerary?: any[]
  recommendations?: any[]
  message?: string
}

function App() {
  const [preferences, setPreferences] = useState<TripPreferences>({
    budget: '',
    duration: '',
    interests: [],
    startLocation: '',
    travelMonth: '',
    optimizationLevel: 'balanced'
  })
  const [showConfirmation, setShowConfirmation] = useState(false)
  const [tripPlan, setTripPlan] = useState<TripPlan | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showMoreRecommendations, setShowMoreRecommendations] = useState(false)

  const interestOptions = [
    { value: 'Beach', label: '🏖️ Beach', desc: 'Coastal and beach destinations' },
    { value: 'Cultural', label: '🏛️ Cultural', desc: 'Temples, heritage sites, historical places' },
    { value: 'Wildlife', label: '🦁 Wildlife', desc: 'National parks, safaris, animal sanctuaries' },
    { value: 'Hiking', label: '🏔️ Hiking', desc: 'Mountains, trekking, scenic trails' },
    { value: 'Peace', label: '🙏 Peace', desc: 'Relaxation, meditation, tranquil spots' },
    { value: 'Party', label: '🎉 Party', desc: 'Nightlife, entertainment, social activities' },
    { value: 'Urban', label: '🏙️ Urban', desc: 'City experiences, modern attractions' },
    { value: 'Adventure', label: '🌄 Adventure', desc: 'Extreme sports, thrilling experiences' }
  ]

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ]

  const optimizationLevels = [
    { value: 'fast', label: '🚀 Fast', desc: 'Quick results (~10 seconds)' },
    { value: 'balanced', label: '⚖️ Balanced', desc: 'Good quality (~25 seconds) [Recommended]' },
    { value: 'thorough', label: '🎯 Thorough', desc: 'Best quality (~60 seconds)' }
  ]

  const startingLocations = [
    'Colombo',
    'Kandy',
    'Galle',
    'Negombo',
    'Jaffna',
    'Trincomalee',
    'Anuradhapura',
    'Polonnaruwa',
    'Nuwara Eliya',
    'Ella',
    'Sigiriya',
    'Bentota',
    'Mirissa',
    'Arugam Bay'
  ]

  const handleInterestToggle = (interest: string) => {
    setPreferences(prev => ({
      ...prev,
      interests: prev.interests.includes(interest)
        ? prev.interests.filter(i => i !== interest)
        : [...prev.interests, interest]
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    const budgetNum = parseFloat(preferences.budget)
    const durationNum = parseInt(preferences.duration)

    if (isNaN(budgetNum) || budgetNum <= 0) {
      setError('Please enter a valid budget greater than 0')
      setLoading(false)
      return
    }

    if (budgetNum < 100 || budgetNum > 2000) {
      setError('Budget must be between $100 and $2000')
      setLoading(false)
      return
    }

    if (isNaN(durationNum) || durationNum < 3 || durationNum > 14) {
      setError('Duration must be between 3 and 14 days')
      setLoading(false)
      return
    }

    if (preferences.interests.length === 0) {
      setError('Please select at least one adventure type')
      setLoading(false)
      return
    }

    setShowConfirmation(true)
    setLoading(false)
  }

  const confirmAndProceed = async () => {
    setShowConfirmation(false)
    setLoading(true)
    setError('')
    
    try {
      const response = await axios.post('http://localhost:8080/api/plan-trip', preferences)
      console.log('API Response:', response.data)
      console.log('Recommendations:', response.data.recommendations)
      setTripPlan(response.data)
      setShowMoreRecommendations(false) // Reset recommendation visibility
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to generate trip plan. Please try again.')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const cancelConfirmation = () => {
    setShowConfirmation(false)
  }

  const handlePlanAnotherTrip = () => {
    setTripPlan(null)
    setShowMoreRecommendations(false)
    setPreferences({
      budget: '',
      duration: '',
      interests: [],
      startLocation: '',
      travelMonth: '',
      optimizationLevel: 'balanced'
    })
    setError('')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleGetMoreRecommendations = () => {
    console.log('Get more recommendations clicked')
    console.log('Current recommendations:', tripPlan?.recommendations)
    console.log('Recommendations length:', tripPlan?.recommendations?.length)
    setShowMoreRecommendations(true)
    setTimeout(() => {
      const recommendationsElement = document.querySelector('.recommendations-section')
      console.log('Recommendations element:', recommendationsElement)
      if (recommendationsElement) {
        recommendationsElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }, 100)
  }

  const getMonthName = (monthNum: string) => {
    if (!monthNum) return 'Current Month'
    const idx = parseInt(monthNum) - 1
    return months[idx] || 'Current Month'
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🌴 Sri Lanka Trip Planner</h1>
          <p>AI-Powered personalized travel itinerary generator</p>
        </header>

        <div className="content">
          <form className="form" onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="budget">Budget (USD)</label>
              <input
                id="budget"
                type="number"
                placeholder="e.g., 500"
                min="100"
                max="2000"
                step="50"
                value={preferences.budget}
                onChange={(e) => setPreferences({ ...preferences, budget: e.target.value })}
                required
              />
              <small className="form-hint">Budget: $100 - $2000 (entrance fees and activities)</small>
            </div>

            <div className="form-group">
              <label htmlFor="duration">Duration (days)</label>
              <input
                id="duration"
                type="number"
                placeholder="e.g., 7"
                min="3"
                max="14"
                value={preferences.duration}
                onChange={(e) => setPreferences({ ...preferences, duration: e.target.value })}
                required
              />
              <small className="form-hint">Enter a value between 3 and 14 days</small>
            </div>

            <div className="form-group">
              <label htmlFor="startLocation">Starting Location</label>
              <select
                id="startLocation"
                value={preferences.startLocation}
                onChange={(e) => setPreferences({ ...preferences, startLocation: e.target.value })}
                required
              >
                <option value="">Select starting location</option>
                {startingLocations.map(location => (
                  <option key={location} value={location}>
                    {location}
                  </option>
                ))}
              </select>
              <small className="form-hint">Choose where your trip will begin</small>
            </div>

            <div className="form-group">
              <label>Travel Month</label>
              <select
                id="travelMonth"
                value={preferences.travelMonth}
                onChange={(e) => setPreferences({ ...preferences, travelMonth: e.target.value })}
              >
                <option value="">Current Month</option>
                {months.map((month, index) => (
                  <option key={month} value={String(index + 1)}>
                    {month}
                  </option>
                ))}
              </select>
              <small className="form-hint">Weather conditions vary by region and season</small>
            </div>

            <div className="form-group">
              <label>Adventure Preferences (Select at least one)</label>
              <div className="interests-grid">
                {interestOptions.map(interest => (
                  <button
                    key={interest.value}
                    type="button"
                    className={`interest-btn ${preferences.interests.includes(interest.value) ? 'active' : ''}`}
                    onClick={() => handleInterestToggle(interest.value)}
                    title={interest.desc}
                  >
                    {interest.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>Optimization Level</label>
              <div className="optimization-options">
                {optimizationLevels.map(level => (
                  <label key={level.value} className="radio-option">
                    <input
                      type="radio"
                      name="optimization"
                      value={level.value}
                      checked={preferences.optimizationLevel === level.value}
                      onChange={(e) => setPreferences({ ...preferences, optimizationLevel: e.target.value })}
                    />
                    <span className="radio-label">
                      <strong>{level.label}</strong>
                      <small>{level.desc}</small>
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? 'Generating Plan...' : 'Generate Trip Plan'}
            </button>
          </form>

          {showConfirmation && (
            <div className="confirmation-dialog">
              <h2>📋 Confirm Your Trip Preferences</h2>
              <div className="confirmation-details">
                <p><strong>📅 Duration:</strong> {preferences.duration} days</p>
                <p><strong>💰 Budget:</strong> ${preferences.budget}</p>
                <p><strong>🎯 Adventures:</strong> {preferences.interests.join(', ')}</p>
                <p><strong>📍 Starting Location:</strong> {preferences.startLocation}</p>
                <p><strong>📆 Travel Month:</strong> {getMonthName(preferences.travelMonth)}</p>
                <p><strong>⚙️ Optimization:</strong> {preferences.optimizationLevel}</p>
              </div>
              <div className="confirmation-buttons">
                <button className="confirm-btn" onClick={confirmAndProceed}>
                  ✓ Proceed with Trip Planning
                </button>
                <button className="cancel-btn" onClick={cancelConfirmation}>
                  ✗ Cancel
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}

          {tripPlan && (
            <div className="results">
              <div className="trip-summary">
                <h2>📋 TRIP SUMMARY</h2>
                <div className="summary-stats">
                  <div className="stat-item">
                    <span className="stat-label">Total Places:</span>
                    <span className="stat-value">{tripPlan.num_places || 0}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Total Cost:</span>
                    <span className="stat-value">${tripPlan.total_cost?.toFixed(2) || '0.00'} / ${tripPlan.user_preferences?.budget || '0'}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Budget Remaining:</span>
                    <span className="stat-value">${tripPlan.budget_remaining?.toFixed(2) || '0.00'}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Total Activity Time:</span>
                    <span className="stat-value">{tripPlan.total_time_hours?.toFixed(1) || '0.0'} hours</span>
                  </div>
                </div>
              </div>

              {tripPlan.schedule && (
                <div className="itinerary-section">
                  <h2>📅 DAY-BY-DAY ITINERARY</h2>
                  {Object.entries(tripPlan.schedule).map(([day, places]: [string, any]) => (
                    <div key={day} className="day-section">
                      <h3>🌅 DAY {day}</h3>
                      {Array.isArray(places) && places.length > 0 ? (
                        <>
                          {places.map((placeName: string, idx: number) => {
                            const placeInfo = tripPlan.itinerary?.find((p: any) => p.name === placeName);
                            if (!placeInfo) return null;
                            return (
                              <div key={idx} className="place-card">
                                <h4>📍 {placeInfo.name}</h4>
                                <div className="place-details">
                                  <p><strong>District:</strong> {placeInfo.district}</p>
                                  <p><strong>Category:</strong> {placeInfo.category}</p>
                                  <p><strong>Adventure:</strong> {placeInfo.adventure_type}</p>
                                  <p><strong>Fee:</strong> ${placeInfo.fee?.toFixed(2)}</p>
                                  <p><strong>Time:</strong> {placeInfo.time?.toFixed(1)} hours</p>
                                </div>
                              </div>
                            );
                          })}
                          <div className="day-summary">
                            <p>💵 Day {day} Cost: ${places.reduce((sum: number, placeName: string) => {
                              const place = tripPlan.itinerary?.find((p: any) => p.name === placeName);
                              return sum + (place?.fee || 0);
                            }, 0).toFixed(2)}</p>
                            <p>⏰ Day {day} Time: {places.reduce((sum: number, placeName: string) => {
                              const place = tripPlan.itinerary?.find((p: any) => p.name === placeName);
                              return sum + (place?.time || 0);
                            }, 0).toFixed(1)} hours</p>
                          </div>
                        </>
                      ) : (
                        <p className="rest-day">Rest day / Travel day</p>
                      )}
                    </div>
                  ))}
                </div>
              )}

              <div className="additional-options">
                <h2>📊 ADDITIONAL OPTIONS</h2>
                <div className="options-buttons">
                  <button 
                    className="option-btn"
                    onClick={handleGetMoreRecommendations}
                  >
                    <span className="option-number">1</span>
                    <span className="option-text">Get more recommendations</span>
                  </button>
                  <button 
                    className="option-btn"
                    onClick={handlePlanAnotherTrip}
                  >
                    <span className="option-number">2</span>
                    <span className="option-text">Plan another trip</span>
                  </button>
                  <button 
                    className="option-btn"
                    onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  >
                    <span className="option-number">3</span>
                    <span className="option-text">Back to top</span>
                  </button>
                </div>
              </div>

              {showMoreRecommendations && (
                <>
                  {tripPlan.recommendations && tripPlan.recommendations.length > 0 ? (
                    <div className="recommendations-section">
                      <h2>🌟 TOP RECOMMENDATIONS</h2>
                      <p className="section-desc">Additional places matching your preferences:</p>
                      <div className="recommendations-grid">
                        {tripPlan.recommendations.map((rec: any, idx: number) => (
                          <div key={idx} className="recommendation-card">
                            <div className="rec-header">
                              <span className="rec-number">{idx + 1}</span>
                              <h4>{rec.name}</h4>
                              <span className="rec-score">Score: {rec.score?.toFixed(3)}</span>
                            </div>
                            <div className="rec-details">
                              <p>📍 {rec.district} | {rec.category}</p>
                              <p>💵 ${rec.fee?.toFixed(1)} | ⏰ {rec.time?.toFixed(1)} hours</p>
                              <p>🎯 {rec.adventure_type}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="recommendations-section">
                      <h2>🌟 TOP RECOMMENDATIONS</h2>
                      <p className="section-desc">No additional recommendations available at this time.</p>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
