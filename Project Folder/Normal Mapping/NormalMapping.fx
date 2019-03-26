//--------------------------------------------------------------------------------------
// File: NormalMapping.fx
//--------------------------------------------------------------------------------------

Texture2D txDiffuse : register(t0);
Texture2D txNormal : register(t1);

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
	matrix World;
	matrix View;
	matrix Projection;

	SurfaceInfo surface;
	Light light;

	float3 EyePosW;
	float HasTexture;
	float HasNormal;
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

	float4 posW = mul(input.posL, World);
	output.posW = posW.xyz;

	output.posH = mul(posW, View);
	output.posH = mul(output.posH, Projection);
	output.tex = input.tex;

	float3 normalW = mul(float4(input.normL, 0.0f), World).xyz;
	output.normW = normalize(normalW);

	output.tangentW = mul(input.tangentL, World);
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
	toEye = mul(toEye, TBN);
	return toEye;
}

//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS(VS_OUTPUT input) : SV_Target
{
	float3 normalW = normalize(input.normW);

	float3 toEye = normalize(EyePosW - input.posW);
	float3 EyeT = EyeVectorToTangentSpace(toEye, normalW, input.tangentW);
	// Get texture data from file
	float4 textureColour = txDiffuse.Sample(samLinear, input.tex);

	float3 lightLecNorm = normalize(light.LightVecW);
	// Compute Colour
	float3 Normal;
	if (HasNormal == 1.0f)
	{
		float4 bumpMap = txNormal.Sample(samLinear, input.tex);

		bumpMap = bumpMap * 2.0f - 1.0f;

		float3 bumpedNormalW = NormalSampleToWorldSpace(bumpMap.xyz, normalW, input.tangentW);
		Normal = bumpedNormalW;

	}
	else if(HasNormal == 0.0f)
	{ 
		Normal = normalW;
	}

	float3 ambient = float3(0.0f, 0.0f, .0f);
	float3 diffuse = float3(0.0f, 0.0f, .0f);
	float3 specular = float3(0.0f, 0.0f, .0f);

	Normal = normalize(Normal);

	// Compute the reflection vector.
	float3 r = reflect(-lightLecNorm, Normal);

	// Determine the diffuse light intensity that strikes the vertex.
	float diffuseAmount = 0.0f;

	// Determine how much specular light makes it into the eye.
	float specularAmount = 0.0f;

	// Only display specular when there is diffuse
	if (!(lightLecNorm.z <= 0.0f) || !(dot(Normal, lightLecNorm) <= 0))
	{
		diffuseAmount = /*surface.DiffuseMtrl * light.DiffuseLight * */dot(Normal, lightLecNorm);
		if (dot(toEye, r) > 0)
		{
			specularAmount = /*surface.SpecularMtrl * light.SpecularLight **/ pow(dot(toEye, r), light.SpecularPower);
		}
	}

	// Compute the ambient, diffuse, and specular terms separately.
	specular += specularAmount * (surface.SpecularMtrl * light.SpecularLight).rgb;
	diffuse += diffuseAmount * (surface.DiffuseMtrl * light.DiffuseLight).rgb;
	ambient += (surface.AmbientMtrl * light.AmbientLight).rgb;

	// Sum all the terms together and copy over the diffuse alpha.
	float4 finalColour;

	if (HasTexture == 1.0f || HasNormal == 1.0f)
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