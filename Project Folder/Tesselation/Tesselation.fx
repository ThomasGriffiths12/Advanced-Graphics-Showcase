//--------------------------------------------------------------------------------------
// File: Tesselation.fx
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

struct HS_IO
{
	float4 pos : SV_POSITION;
	float3 worldPos : POSITION;
	float3 norm : NORMAL;
	float2 tex : TEXCOORD0;
	float3 tangentW : TANGENT;
};

//--------------------------------------------------------------------------------------
struct PS_INPUT
{
	float4 posCS : POSITION2;
    float4 posH : SV_POSITION;
	float3 normW : NORMAL;
	float3 tangentW : TANGENT;
	float3 posW : POSITION;
	float2 tex : TEXCOORD0;
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
HS_IO VS(VS_INPUT input)
{
	HS_IO output;
	float4 worldPos = mul(input.posL, world);
	output.worldPos = worldPos.xyz;

	output.pos = input.posL;

	//float4 posW = mul(input.posL, world);
	//output.posW = posW.xyz;
	//output.posH = mul(posW, view);
	//output.posH = mul(output.posH, projection);

	/*float3x3 tbnMatrix;*/
	//tbnMatrix[0] = normalize(mul(float4(input.tangentL, 0.0), World).xyz);
	//tbnMatrix[1] = normalize(mul(float4(cross(input.normL, input.tangentL), 0.0f), World).xyz);
	output.tex = input.tex;

	float3 normalW = mul(float4(input.normL, 0.0f), world).xyz;
	output.norm = normalize(normalW);

	output.tangentW = mul(float4(input.tangentL, 0.0f), world).xyz;
    return output;
}

[domain("tri")]
[partitioning("fractional_even")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PassThroughConstantHS")]
[maxtessfactor(64.0)]
HS_IO HSMAIN(
	InputPatch<HS_IO, 3> ip,
	uint i : SV_OutputControlPointID,
	uint PatchID : SV_PrimitiveID)
{
	HS_IO output;
	output.pos = ip[i].pos;
	output.worldPos = ip[i].worldPos;
	output.norm = ip[i].norm;
	output.tex = ip[i].tex;
	output.tangentW = ip[i].tangentW;
	return output;
}

struct HS_CONSTANT_DATA_OUTPUT
{
	float Edges[3]			: SV_TessFactor;
	float Inside : SV_InsideTessFactor;
};

HS_CONSTANT_DATA_OUTPUT PassThroughConstantHS(InputPatch<HS_IO, 3> ip,
	uint PatchID : SV_PrimitiveID)
{
	float tessellationFactor = 32;
	HS_CONSTANT_DATA_OUTPUT output;
	output.Edges[0] = tessellationFactor;
	output.Edges[1] = tessellationFactor;
	output.Edges[2] = tessellationFactor;
	output.Inside = tessellationFactor;
	return output;
}
//Domain Shader
[domain("tri")]
PS_INPUT DSMAIN(HS_CONSTANT_DATA_OUTPUT input,
	float3 BarycentricCoordinates : SV_DomainLocation,
	const OutputPatch<HS_IO, 3> TrianglePatch)
{
	PS_INPUT output;

	// Interpolate world space position with barycentric coordinates
	float3 vWorldPos = BarycentricCoordinates.x * TrianglePatch[0].pos
		+ BarycentricCoordinates.y * TrianglePatch[1].pos
		+ BarycentricCoordinates.z * TrianglePatch[2].pos;

	output.posH = float4(vWorldPos.xyz, 1.0);

	float2 vTex = BarycentricCoordinates.x * TrianglePatch[0].tex
		+ BarycentricCoordinates.y * TrianglePatch[1].tex
		+ BarycentricCoordinates.z * TrianglePatch[2].tex;

	output.tex = vTex;

	output.normW = BarycentricCoordinates.x * TrianglePatch[0].norm
		+ BarycentricCoordinates.y * TrianglePatch[1].norm
		+ BarycentricCoordinates.z * TrianglePatch[2].norm;

	output.normW = normalize(output.normW);

	float3 tangentW = BarycentricCoordinates.x * TrianglePatch[0].tangentW
		+ BarycentricCoordinates.y * TrianglePatch[1].tangentW
		+ BarycentricCoordinates.z * TrianglePatch[2].tangentW;

	output.tangentW = tangentW;

	vWorldPos = BarycentricCoordinates.x * TrianglePatch[0].worldPos
		+ BarycentricCoordinates.y * TrianglePatch[1].worldPos
		+ BarycentricCoordinates.z * TrianglePatch[2].worldPos;
	output.posW = (vWorldPos.xyz);

	const float mipInterval = 20.0f;	float mipLevel = clamp((distance(output.posW, eyePosW) - mipInterval) / mipInterval, 0.0f, 6.0f);
	float displacement = txDisplace.SampleLevel(samLinear, output.tex, mipLevel).r;
	displacement *= scale;
	displacement += bias;
	float3 direction = -output.normW;
	output.posW += direction * displacement;

	// Need to do WVP here not in VS!
	output.posH = mul(output.posH, world);
	output.posH = mul(output.posH, view);
	output.posH = mul(output.posH, projection);
	//output.posH *= float4(output.posW.xyz, 1.0);
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
	//float3 bumpedNormalW = mul(normalMapSample, TBN);
	float3 bumpedNormalW = mul(normalMapSample,TBN);
	return bumpedNormalW;
}

float3 EyeVectorToTangentSpace(float3 toEye, float3 unitNormalW, float3 tangentW)
{
	float3 N = unitNormalW;
	float3 T = normalize(tangentW - dot(tangentW, N) *N);
	float3 B = normalize(cross(N, T));
	float3x3 TBN = float3x3(T, B, N);
	TBN = transpose(TBN);
	//toEye = normalize(mul(TBN, toEye));
	toEye = mul(toEye, TBN);

	return toEye;
}
float3 LightVectorToTangentSpace(float3 lightLec, float3 unitNormalW, float3 tangentW)
{
	float3 N = unitNormalW;
	float3 T = normalize(tangentW - dot(tangentW, N) *N);
	float3 B = normalize(cross(N, T));
	float3x3 TBN = float3x3(T, B, N);
	TBN = transpose(TBN);
	//lightLec = normalize(mul(TBN, lightLec));
	lightLec = mul(lightLec, TBN);
	return lightLec;
}
float3 WorldToTangent(float3 normalW, float3 tangentW)
{
	float3 N = normalW;
	float3 T = normalize(tangentW - dot(tangentW, N) * N);
	float3 B = normalize(cross(N, T));
	float3x3 TBN = float3x3(T, B, N);
	TBN = transpose(TBN);
	//float3 normalT = normalize(mul(TBN, normalW));
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
			float delta2 = ( + fStepSize) - fLastSampledHeight;

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
float4 PS(PS_INPUT input) : SV_Target
{
	float3 normalW = normalize(input.normW);

	float3 toEye = normalize(eyePosW - input.posW);

	//float2 newCoords = ParallaxMapping(input.tex, toEye, bias, scale, normalW, input.tangentW);
	float2 newCoords = ParallaxOcclusionMapping(input.tex, toEye, bias, scale, normalW, input.tangentW, maxSamples, minSamples);
	//float2 newCoords = input.tex;
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
	float specularAmount = 0.0f;

	// Determine the diffuse light intensity that strikes the vertex.
	float diffuseAmount = 0.0f;

	// Only display specular when there is diffuse
	if (((lightLecNorm.z <= 0.0f) || (dot(Normal, lightLecNorm) <= 0)))
	{
		diffuseAmount = dot(Normal, lightLecNorm);
		if (dot(toEye, r) > 0)
		{
			specularAmount = pow(dot(toEye, r), light.SpecularPower);
		}
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