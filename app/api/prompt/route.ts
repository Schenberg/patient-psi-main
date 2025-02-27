import 'server-only'

import { NextRequest } from 'next/server';
import { NextResponse } from 'next/server';
import { setProfile, setPatientType, sampleProfile } from '@/app/api/getDataFromKV'

export const dynamic = 'force-dynamic'

export async function POST(request: Request) {
    try {
        const data = await request.json();
        await setPatientType(data.patientType);
        return NextResponse.json({ message: 'Patient type submitted successfully' });
    } catch (error) {
        console.error('Error in POST /api/prompt:', error);
        return NextResponse.json({ error: 'Internal Server Error in POST' }, { status: 500 });
    }
}

export async function GET(request: NextRequest) {
    try {
        const profile = await sampleProfile();
        if (!profile) {
            throw new Error('No profile returned from sampleProfile');
        }
        await setProfile(profile);
        return NextResponse.json({ profile });
    } catch (error) {
        console.error('Error in GET /api/prompt:', error);
        return NextResponse.json({ error: 'Internal Server Error in GET' }, { status: 500 });
    }
}
