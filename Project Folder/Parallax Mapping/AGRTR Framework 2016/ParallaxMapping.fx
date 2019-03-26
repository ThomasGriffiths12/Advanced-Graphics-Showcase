//--------------------------------------------------------------------------------------
// File: ParallaxMapping.fx
//--------------------------------------------------------------------------------------

Texture2D txDiffuse : register(t0);
Texture2D txNormal : register(t1);
Texture2D txDisplace : register(t2);

SamplerState samLinear : register(s0)
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = WRAP;
	AddressV = WRAP;
};

//--------------------------------------------------------------------------------------
// Constant Buffer Variables
//--------------------------------------------------------------------------------------

struct SurfaceInfo
{
	float4 AmbientMtrl;
	float4 DiffuseMtrl;
	float4 SpecularMtrl;
};

struct Light
{
	float4 AmbientLight;
	float4 DiffuseLight;
	float4 SpecularLight;

	float SpecularPower;
	float3 LightVecW;
};

cbuffer ConstantBuffer : register( b0 )
{
	matrix world;
	matrix view;
	matrix projection;

	SurfaceInfo surface;
	Light light;

	float3 eyePosW;
	float hasTexture;
	float hasNormal;

	float scale;
	float bias;

	int minSamples;
	int maxSamples;
}

struct VS_INPUT
{
	float4 posL : POSITION;
	float3 normL : NORMAL;
	float2 tex : TEXCOORD0;
	float3 tangentL : TANGENT;
};

//--------------------------------------------------------------------------------------
struct VS_OUTPUT
{
    float4 posH : SV_POSITION;
	float3 normW : NORMAL;
	float3 tangentW : TANGENT;
	float3 posW : POSITION;
	float2 tex : TEXCOORD0;
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
VS_OUTPUT VS(VS_INPUT input)
{
    VS_OUTPUT output = (VS_OUTPUT)0;

	float4 posW = mul(input.posL, world);
	output.posW = posW.xyz;
	output.posH = mul(posW, view);
	output.posH = mul(output.posH, projection);

	/*float3x3 tbnMatrix;*/
	//tbnMatrix[0] = normalize(mul(float4(input.tangentL, 0.0), World).xyz);
	//tbnMatrix[1] = normalize(mul(float4(cross(input.normL, input.tangentL), 0.0f), World).xyz);
	output.tex = input.tex;

	float3 normalW = mul(float4(input.normL, 0.0f), world).xyz;
	output.normW = normalize(normalW);

	output.tangentW = mul(float4(input.tangentL, 0.0f), world).xyz;
    return output;
}

//--------------------------------------------------------------------------------------
// Normal Sampler
//--------------------------------------------------------------------------------------
float3 NormalSampleToWorldSpace(float3 normalMapSample, float3 unitNormalW, float3 tangentW)
{
	//Build Tangent, Binormal, Normal matrix
	float3 N = unitNormalW;
	float3 T = normalize(tangentW - dot(tangentW, N) *N);
	float3 B = normalize(cross(N, T));
	float3x3 TBN = float3x3(T, B, N);

	//Transform from tangent space to world space.
	float3 bumpedNormalW = mul(normalMapSample, TBN);
	return bumpedNormalW;
}

float3 EyeVectorToTangentSpace(float3 toEye, float3 unitNormalW, float3 tangentW)
{
	float3 N = unitNormalW;
	float3 T = normalize(tangentW - dot(tangentW, N) *N);
	float3 B = normalize(cross(N, T));
	float3x3 TBN = float3x3(T, B, N);
	TBN = transpose(TBN);
	toEye = normalize(mul(toEye, TBN));
	return toEye;
}

float3 WorldToTangent(float3 normalW, float3 tangentW)
{
	float3 N = normalW;
	float3 T = normalize(tangentW - dot(tangentW, N) * N);
	float3 B = normalize(cross(N, T));
	float3x3 TBN = float3x3(T, B, N);
	TBN = transpose(TBN);
	float3 normalT = normalize(mul(normalW, TBN));
	return normalT;
}
float2 ParallaxMapping(float2 oldUV, float3 toEye, float bias, float scale, float3 normalW, float3 tangentW)
{
	toEye = EyeVectorToTangentSpace(toEye, normalW, tangentW);

	float displacement = txDisplace.Sample(samLinear, oldUV).r;

	float heightMapScale = (displacement * scale) + bias;

	float2 newCoords = oldUV + (normalize(-toEye) * heightMapScale);

	return newCoords;
}

float2 ParallaxOcclusionMapping(float2 oldUV, float3 toEye, float bias, float scale, float3 normalW, float3 tangentW, int nMaxSamples, int nMinSamples)
{
	float3 toEyeT = EyeVectorToTangentSpace(-toEye, normalize(normalW), tangentW);
	float3 normalT = WorldToTangent(normalize(normalW), tangentW);
	float displacement = txDisplace.Sample(samLinear, oldUV).r;
	float heightMapScale = (displacement * scale) + bias;

	float fParallaxLimit = -length(toEyeT.xy) / toEyeT.z;
	fParallaxLimit *= heightMapScale;
	float2 vOffsetDir = normalize(toEyeT.xy);
	float2 vMaxOffset = -vOffsetDir * fParallaxLimit;

	int nNumSamples = (int)lerp(nMaxSamples, nMinSamples, dot(toEye, normalW));
	float fStepSize = 1.0 / (float)nNumSamples;

	float2 dx = ddx(oldUV);
	float2 dy = ddy(oldUV);

	float fCurrRayHeight = 1.0 - fStepSize;
	float fPrevRayHeight = 1.0;

	float2 vCurrOffset = float2(0, 0);
	float2 vLastOffset = float2(0, 0);

	float fLastSampledHeight = 0.0f;
	float fCurrSampledHeight = 0.0f;

	int nCurrSample = 0;

	while (nCurrSample < nNumSamples + 1)
	{
		fCurrSampledHeight = txDisplace.SampleGrad(samLinear, oldUV + vCurrOffset, dx, dy).r;
		if (fCurrSampledHeight > fCurrRayHeight)
		{
			float delta2 = (+fStepSize) - fLastSampledHeight;

			float ratio = fCurrSampledHeight - fCurrRayHeight / (fLastSampledHeight - fCurrRayHeight + fPrevRayHeight);

			vCurrOffset = vLastOffset + ratio * (fStepSize * vCurrOffset);

			nCurrSample = nNumSamples + 1;
		}
		else
		{
			nCurrSample++;

			fPrevRayHeight = fCurrRayHeight;
			vLastOffset = vCurrOffset;
			fLastSampledHeight = fCurrSampledHeight;

			vCurrOffset += fStepSize * vMaxOffset;
			fCurrRayHeight -= fStepSize;
		}
	}
	float2 vFinalCoords = oldUV + vCurrOffset;
	return vFinalCoords;
}

//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS(VS_OUTPUT input) : SV_Target
{
	float3 normalW = normalize(input.normW);

	float3 toEye = normalize(eyePosW - input.posW);

	//float2 newCoords = ParallaxMapping(input.tex, toEye, bias, scale, normalW, input.tangentW);
	float2 newCoords = ParallaxOcclusionMapping(input.tex, toEye, bias, scale, normalW, input.tangentW, maxSamples, minSamples);

	float4 textureColour = txDiffuse.Sample(samLinear, newCoords);

	float3 ambient = float3(0.0f, 0.0f, .0f);
	float3 diffuse = float3(0.0f, 0.0f, .0f);
	float3 specular = float3(0.0f, 0.0f, .0f);

	float3 lightLecNorm = normalize(light.LightVecW);
	// Compute Colour
	float3 Normal;
	if (hasNormal == 1.0f)
	{
		float4 bumpMap = txNormal.Sample(samLinear, newCoords);

		bumpMap = bumpMap * 2.0f - 1.0f;

		float3 bumpedNormalW = NormalSampleToWorldSpace(bumpMap.xyz, normalW, input.tangentW);
		Normal = bumpedNormalW;

	}
	else if(hasNormal == 0.0f)
	{ 
		Normal = normalW;
	}

	Normal = normalize(Normal);

	// Compute the reflection vector.
	float3 r = reflect(-lightLecNorm, Normal);

	// Determine how much specular light makes it into the eye.
	float specularAmount = pow(max(dot(r, toEye), 0.0f), light.SpecularPower);

	// Determine the diffuse light intensity that strikes the vertex.
	float diffuseAmount = max(dot(lightLecNorm, Normal), 0.0f);

	// Only display specular when there is diffuse
	if (diffuseAmount <= 0.0f)
	{
		specularAmount = 0.0f;
	}

	// Compute the ambient, diffuse, and specular terms separately.
	specular += specularAmount * (surface.SpecularMtrl * light.SpecularLight).rgb;
	diffuse += diffuseAmount * (surface.DiffuseMtrl * light.DiffuseLight).rgb;
	ambient += (surface.AmbientMtrl * light.AmbientLight).rgb;

	// Sum all the terms together and copy over the diffuse alpha.
	float4 finalColour;

	if (hasTexture == 1.0f || hasNormal == 1.0f)
	{
		finalColour.rgb = (textureColour.rgb * (ambient + diffuse)) + specular;
	}
	else
	{
		finalColour.rgb = ambient + diffuse + specular;
	}

	finalColour.a = surface.DiffuseMtrl.a;

	return finalColour;
}