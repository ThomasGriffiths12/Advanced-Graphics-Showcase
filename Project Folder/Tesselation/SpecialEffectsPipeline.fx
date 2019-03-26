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

struct VS_INPUT
{
	float4 posL : POSITION;
	float2 tex : TEXCOORD0;
};

//--------------------------------------------------------------------------------------
struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
	float2 tex : TEXCOORD0;
};

//Changes the colour of the render target to negative
float4 ChangeColourToNegative(float4 textureColour)
{
	float4 negativeColour = textureColour;
	negativeColour.rgb = 1 - negativeColour.rgb;
	return negativeColour;
}

float4 BlurTextureColour(float2 texCoord, float offset)
{
	float4 blurredColour = txDiffuse.Sample(samLinear, float2(texCoord.x + offset, texCoord.y + offset));
	blurredColour += txDiffuse.Sample(samLinear, float2(texCoord.x - offset, texCoord.y - offset));
	blurredColour += txDiffuse.Sample(samLinear, float2(texCoord.x + offset, texCoord.y - offset));
	blurredColour += txDiffuse.Sample(samLinear, float2(texCoord.x - offset, texCoord.y + offset));

	blurredColour = blurredColour / 4;
	return blurredColour;
}
//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
VS_OUTPUT VS(VS_INPUT input)
{
    VS_OUTPUT output = (VS_OUTPUT)0;

	output.pos = input.posL;
	output.tex = input.tex;

    return output;
}

//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS(VS_OUTPUT input) : SV_Target
{
	//return float4(input.tex, 0.0f, 1.0f);
	float4 textureColour = txDiffuse.Sample(samLinear, input.tex);

	float offset = 0.0008f;
	//textureColour = BlurTextureColour(input.tex, offset);
	//textureColour = ChangeColourToNegative(textureColour);
	return textureColour;
}