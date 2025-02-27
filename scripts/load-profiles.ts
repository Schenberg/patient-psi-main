import { kv } from '@vercel/kv'
import fs from 'fs'
import path from 'path'

async function loadProfiles() {
    try {
        const profilesPath = path.join(process.cwd(), 'python', 'data', 'profiles.json')
        const profilesData = fs.readFileSync(profilesPath, 'utf8')
        const profiles = JSON.parse(profilesData)

        for (const profile of profiles) {
            const key = `profile_${profile.id}`
            await kv.set(key, profile)
            console.log(`Loaded profile: ${key}`)
        }

        console.log('All profiles loaded successfully')
    } catch (error) {
        console.error('Error loading profiles:', error)
    }
}

loadProfiles()
